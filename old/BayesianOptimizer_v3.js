/**
 * 贝叶斯优化器 (增强版) - 用于超参数优化的贝叶斯优化实现
 *
 * @param {Object} cfg 配置对象
 * @param {Object} cfg.searchSpace 搜索空间定义。
 *   格式: {
 *     paramName1: { type: 'cont', range: [min, max] }, // 连续值 (默认)
 *     paramName2: { type: 'int', range: [min, max] },  // 整数
 *     paramName3: { type: 'cat', values: [val1, val2] } // 分类值
 *   }
 * @param {Function} cfg.objective 目标函数，接受参数对象和迭代索引，返回Promise<number>
 * @param {number} cfg.iterations 最大迭代次数，默认50
 * @param {number} cfg.parallel 并行评估数量，默认1
 * @param {number|null} cfg.seed 随机数种子，默认null
 * @param {boolean} cfg.debug 是否输出调试信息，默认false
 * @param {string} cfg.acquisitionFunction 采集函数类型，支持'EI'、'UCB'、'PI'，默认'EI'
 * @param {number} cfg.maxObservations 最大观测点数量，用于限制内存使用，默认100
 * @returns {Promise<Object>} 优化结果，包含bestParams和bestLoss
 */
export const BayesianOptimizer = async (cfg) => {
	const {
		searchSpace,
		objective,
		iterations = 50,
		parallel = 1,
		seed = null,
		debug = false,
		acquisitionFunction = 'EI',
		maxObservations = 100
	} = cfg;

	// --- 辅助函数与工具 ---

	const rng = (() => {
		let state = seed !== null ? seed : Math.random() * 1000000;
		return () => {
			state = (state * 9301 + 49297) % 233280;
			return state / 233280;
		};
	})();

	const erf = (x) => {
		const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
		const a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
		const sign = x < 0 ? -1 : 1;
		x = Math.abs(x);
		const t = 1.0 / (1.0 + p * x);
		const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
		return sign * y;
	};

	const pdf = (x) => Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
	const cdf = (x) => 0.5 * (1 + erf(x / Math.SQRT2));

	// --- 参数处理器 (核心改进：支持混合类型) ---
	const parameterProcessor = (() => {
		const paramKeys = Object.keys(searchSpace);
		const specs = paramKeys.map(key => {
			const def = searchSpace[key];
			if (Array.isArray(def)) return { key, type: 'cont', range: def }; // 兼容旧格式
			return { key, ...def };
		});

		const normalizedDim = specs.reduce((sum, s) => {
			if (s.type === 'cat') return sum + s.values.length;
			return sum + 1;
		}, 0);

		const normalize = (params) => {
			const normalized = new Array(normalizedDim).fill(0);
			let idx = 0;
			for (const spec of specs) {
				const value = params[spec.key];
				if (spec.type === 'int' || spec.type === 'cont') {
					const [min, max] = spec.range;
					normalized[idx++] = (Math.max(min, Math.min(max, value)) - min) / (max - min);
				} else if (spec.type === 'cat') {
					const catIndex = spec.values.indexOf(value);
					if (catIndex !== -1) {
						normalized[idx + catIndex] = 1;
					}
					idx += spec.values.length;
				}
			}
			return normalized;
		};

		const denormalize = (normalized) => {
			const params = {};
			let idx = 0;
			for (const spec of specs) {
				if (spec.type === 'int' || spec.type === 'cont') {
					const [min, max] = spec.range;
					let val = normalized[idx++] * (max - min) + min;
					if (spec.type === 'int') {
						val = Math.round(val);
					}
					params[spec.key] = val;
				} else if (spec.type === 'cat') {
					const oneHotVector = normalized.slice(idx, idx + spec.values.length);
					const catIndex = oneHotVector.indexOf(Math.max(...oneHotVector));
					params[spec.key] = spec.values[catIndex];
					idx += spec.values.length;
				}
			}
			return params;
		};

		return { normalize, denormalize, specs, normalizedDim };
	})();

	// 拉丁超立方采样 (LHS)
	const generateLHSSamples = (numSamples) => {
		const dims = parameterProcessor.specs.length;
		const grid = Array(dims).fill(0).map(() => {
			const divisions = Array(numSamples).fill(0).map((_, i) => (i + rng()) / numSamples);
			for (let i = numSamples - 1; i > 0; i--) {
				const j = Math.floor(rng() * (i + 1));
				[divisions[i], divisions[j]] = [divisions[j], divisions[i]];
			}
			return divisions;
		});

		const samples = [];
		for (let i = 0; i < numSamples; i++) {
			const tempNormalized = {};
			let gridIdx = 0;
			for (const spec of parameterProcessor.specs) {
				if (spec.type === 'int' || spec.type === 'cont') {
					tempNormalized[spec.key] = grid[gridIdx++][i];
				} else if (spec.type === 'cat') {
					const catIndex = Math.floor(grid[gridIdx++][i] * spec.values.length);
					tempNormalized[spec.key] = spec.values[catIndex];
				}
			}
			samples.push(parameterProcessor.denormalize(parameterProcessor.normalize(tempNormalized)));
		}
		return samples;
	};
    
	// --- 高斯过程 (GP) 相关 ---

	// RBF 核函数 (核心改进：支持ARD)
	const kernel = (x1, x2, lengthScales) => {
		let sum = 0;
		for (let i = 0; i < x1.length; i++) {
			const diff = x1[i] - x2[i];
			const scale = lengthScales[i];
			sum += (diff * diff) / (scale * scale);
		}
		return Math.exp(-0.5 * sum);
	};

	const computeKernelMatrix = (observations, lengthScales, noiseLevel) => {
		const n = observations.length;
		const K = Array(n).fill(0).map(() => Array(n).fill(0));
		for (let i = 0; i < n; i++) {
			for (let j = 0; j < n; j++) {
				K[i][j] = kernel(observations[i].normalized, observations[j].normalized, lengthScales);
			}
			K[i][i] += noiseLevel;
		}
		return K;
	};

	// Cholesky 分解 (核心改进：增加数值稳定性)
	const choleskyDecomposition = (A) => {
		const n = A.length;
		const L = Array(n).fill(0).map(() => Array(n).fill(0));
		for (let i = 0; i < n; i++) {
			for (let j = 0; j <= i; j++) {
				let sum = 0;
				for (let k = 0; k < j; k++) {
					sum += L[i][k] * L[j][k];
				}
				if (i === j) {
					const diagValue = A[i][i] - sum;
					if (diagValue < 1e-10) {
						throw new Error("Matrix not positive definite");
					}
					L[i][j] = Math.sqrt(diagValue);
				} else {
					L[i][j] = (A[i][j] - sum) / L[j][j];
				}
			}
		}
		return L;
	};

	const solveTriangular = (L, b, lower = true) => {
		const n = L.length;
		const x = Array(n).fill(0);
		if (lower) {
			for (let i = 0; i < n; i++) {
				let sum = 0;
				for (let j = 0; j < i; j++) {
					sum += L[i][j] * x[j];
				}
				x[i] = (b[i] - sum) / L[i][i];
			}
		} else {
			for (let i = n - 1; i >= 0; i--) {
				let sum = 0;
				for (let j = i + 1; j < n; j++) {
					sum += L[j][i] * x[j];
				}
				x[i] = (b[i] - sum) / L[i][i];
			}
		}
		return x;
	};

	// GP超参数优化器 (核心改进：坐标上升法替代网格搜索)
	const hyperparameterOptimizer = (observations) => {
		const y = observations.map(obs => obs.loss);
		const n = observations.length;
		const defaultNoise = 1e-6;
		
		if (n < 3) {
			return { 
				lengthScales: new Array(parameterProcessor.normalizedDim).fill(0.5), 
				noiseLevel: defaultNoise 
			};
		}

		let bestLengthScales = new Array(parameterProcessor.normalizedDim).fill(0.5);
		let bestLogLikelihood = -Infinity;

		// 使用坐标上升法优化 lengthScales
		const optimizationIterations = 5;
		const candidates = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5];

		let currentLengthScales = [...bestLengthScales];

		for (let iter = 0; iter < optimizationIterations; iter++) {
			for (let dim = 0; dim < parameterProcessor.normalizedDim; dim++) {
				let bestDimScale = currentLengthScales[dim];
				let bestDimLikelihood = -Infinity;

				for (const scale of candidates) {
					const tempScales = [...currentLengthScales];
					tempScales[dim] = scale;
					
					try {
						const K = computeKernelMatrix(observations, tempScales, defaultNoise);
						const L = choleskyDecomposition(K);
						const alpha = solveTriangular(L, solveTriangular(L, y, true), false);
						let logDet = 0;
						for (let i = 0; i < n; i++) logDet += 2 * Math.log(L[i][i]);
						const dataFit = y.reduce((s, val, i) => s + val * alpha[i], 0);
						const logLikelihood = -0.5 * dataFit - 0.5 * logDet;

						if (logLikelihood > bestDimLikelihood) {
							bestDimLikelihood = logLikelihood;
							bestDimScale = scale;
						}
					} catch (e) { continue; }
				}
				currentLengthScales[dim] = bestDimScale;
			}

			// 检查整体似然是否改善
			try {
				const K = computeKernelMatrix(observations, currentLengthScales, defaultNoise);
				const L = choleskyDecomposition(K);
				const alpha = solveTriangular(L, solveTriangular(L, y, true), false);
				let logDet = 0;
				for (let i = 0; i < n; i++) logDet += 2 * Math.log(L[i][i]);
				const dataFit = y.reduce((s, val, i) => s + val * alpha[i], 0);
				const currentLogLikelihood = -0.5 * dataFit - 0.5 * logDet;

				if (currentLogLikelihood > bestLogLikelihood) {
					bestLogLikelihood = currentLogLikelihood;
					bestLengthScales = [...currentLengthScales];
				}
			} catch(e) { /* ignore */ }
		}

		if (debug) {
			console.log(`Optimized Length Scales: [${bestLengthScales.map(s => s.toFixed(2)).join(', ')}]`);
		}
		
		return { lengthScales: bestLengthScales, noiseLevel: defaultNoise };
	};

	const gpPredict = (observations, xNew, hyperparams) => {
		const n = observations.length;
		let K, L;
		let noise = hyperparams.noiseLevel;
		
		// 鲁棒性处理：如果Cholesky分解失败，增加噪声重试
		for (let i = 0; i < 5; i++) {
			try {
				K = computeKernelMatrix(observations, hyperparams.lengthScales, noise);
				L = choleskyDecomposition(K);
				break;
			} catch (e) {
				if (i === 4) throw e; // 尝试多次后仍然失败
				noise *= 10;
			}
		}

		const kStar = observations.map(obs => kernel(obs.normalized, xNew, hyperparams.lengthScales));
		const y = observations.map(obs => obs.loss);
		
		const alpha = solveTriangular(L, solveTriangular(L, y, true), false);
		const mean = kStar.reduce((sum, val, i) => sum + val * alpha[i], 0);
		
		const v = solveTriangular(L, kStar, true);
		let variance = kernel(xNew, xNew, hyperparams.lengthScales);
		variance = v.reduce((s, val) => s - val * val, variance);

		return { mean, variance: Math.max(variance, 0) };
	};

	// --- 采集与候选点选择 ---

	const acquisitionFunctions = {
		EI: (mean, std, bestLoss, xi = 0.01) => {
			if (std < 1e-9) return 0;
			const gamma = (bestLoss - mean - xi) / std;
			return std * (gamma * cdf(gamma) + pdf(gamma));
		},
		UCB: (mean, std, bestLoss, kappa = 2.576) => mean - kappa * std, // 最小化问题，所以是减号
		PI: (mean, std, bestLoss, xi = 0.01) => {
			if (std < 1e-9) return 0;
			const gamma = (bestLoss - mean - xi) / std;
			return cdf(gamma);
		}
	};
    
	// 候选点查找器 (核心改进：锦标赛选择演化策略)
	const candidateFinder = (observations, bestLoss, numCandidates, hyperparams) => {
		if (observations.length < 5) {
			return Array(numCandidates).fill(0).map(() => parameterProcessor.denormalize(
				Array(parameterProcessor.normalizedDim).fill(0).map(() => rng())
			));
		}

		const acquisitionFunc = acquisitionFunctions[acquisitionFunction] || acquisitionFunctions.EI;
		const populationSize = 50;
		const evolutionSteps = 20;

		// 1. 初始化种群
		let population = Array(populationSize).fill(0).map(() => {
			const normalized = Array(parameterProcessor.normalizedDim).fill(0).map(() => rng());
			const { mean, variance } = gpPredict(observations, normalized, hyperparams);
			const acqValue = acquisitionFunc(mean, Math.sqrt(variance), bestLoss);
			return { normalized, acqValue };
		});

		// 2. 演化迭代
		for (let step = 0; step < evolutionSteps; step++) {
			// 锦标赛选择
			const parent1 = population[Math.floor(rng() * populationSize)];
			const parent2 = population[Math.floor(rng() * populationSize)];
			const winner = parent1.acqValue > parent2.acqValue ? parent1 : parent2;

			// 变异
			const offspringNormalized = winner.normalized.map(val => {
				let mutated = val + (rng() - 0.5) * 0.1; // 扰动
				return Math.max(0, Math.min(1, mutated));
			});

			const { mean, variance } = gpPredict(observations, offspringNormalized, hyperparams);
			const offspringAcqValue = acquisitionFunc(mean, Math.sqrt(variance), bestLoss);

			// 替换最差的个体
			let worstIdx = 0;
			for(let i = 1; i < populationSize; i++) {
				if(population[i].acqValue < population[worstIdx].acqValue) {
					worstIdx = i;
				}
			}
			if (offspringAcqValue > population[worstIdx].acqValue) {
				population[worstIdx] = { normalized: offspringNormalized, acqValue: offspringAcqValue };
			}
		}

		// 3. 选出最优的 N 个
		return population
			.sort((a, b) => b.acqValue - a.acqValue)
			.slice(0, numCandidates)
			.map(p => parameterProcessor.denormalize(p.normalized));
	};

	const selectMostInformativeObservations = (observations, maxSize) => {
		if (observations.length <= maxSize) return observations;
		const sorted = [...observations].sort((a, b) => a.loss - b.loss);
		const bestCount = Math.floor(maxSize * 0.3);
		const randomCount = maxSize - bestCount;
		const selected = sorted.slice(0, bestCount);
		const remaining = sorted.slice(bestCount);
		for (let i = 0; i < randomCount && remaining.length > 0; i++) {
			const randomIndex = Math.floor(rng() * remaining.length);
			selected.push(remaining.splice(randomIndex, 1)[0]);
		}
		return selected;
	};

	// --- 主优化循环 ---

	let observations = [];
	let bestLoss = Infinity;
	let bestParams = null;
	let iteration = 0;

	// 初始化采样
	const initialSamples = Math.max(5, parallel * 2);
	const initialParamsList = generateLHSSamples(initialSamples);
	
	for (let i = 0; i < initialSamples; i++) {
		const params = initialParamsList[i];
		const loss = await objective(params, i);
		observations.push({ params, normalized: parameterProcessor.normalize(params), loss });
		if (loss < bestLoss) {
			bestLoss = loss;
			bestParams = params;
		}
		if (debug) console.log(`[Initial ${i}] Loss: ${loss.toFixed(4)}, Best: ${bestLoss.toFixed(4)}`);
	}
	iteration = initialSamples;

	// 贝叶斯优化主循环
	while (iteration < iterations) {
		const hyperparams = hyperparameterOptimizer(observations);
		const candidates = candidateFinder(observations, bestLoss, parallel, hyperparams);
		
		const promises = candidates.map((params, i) => {
			if (iteration + i >= iterations) return null;
			const idx = iteration + i;
			return (async () => {
				const loss = await objective(params, idx);
				const normalized = parameterProcessor.normalize(params);
				const newObservation = { params, normalized, loss };
				
				// 并行环境下需要线程安全地更新
				// 在JS的Promise.all模型下，可以暂时在最后统一处理
				if (debug) console.log(`[${idx}] Loss: ${loss.toFixed(4)}, Best: ${bestLoss.toFixed(4)}`);
				
				return newObservation;
			})();
		}).filter(p => p !== null);

		const newObservations = await Promise.all(promises);
		
		// 更新观测点和最优解
		for (const obs of newObservations) {
			observations.push(obs);
			if (obs.loss < bestLoss) {
				bestLoss = obs.loss;
				bestParams = obs.params;
				if (debug) console.log(`[${iteration + newObservations.indexOf(obs)}] New best: ${bestLoss.toFixed(4)}`);
			}
		}
		
		iteration += newObservations.length;

		if (observations.length > maxObservations) {
			observations = selectMostInformativeObservations(observations, maxObservations);
		}
	}

	return { bestParams, bestLoss, totalIterations: iteration };
};

export default BayesianOptimizer;
