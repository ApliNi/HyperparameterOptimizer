/**
 * 贝叶斯优化器 (平衡进化版) - 解决过早收敛问题
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

	// --- 随机数生成器 ---
	const rng = (() => {
		let state = seed !== null ? seed : Math.random() * 1000000;
		return () => {
			state = (state * 9301 + 49297) % 233280;
			return state / 233280;
		};
	})();

	// --- 数学工具函数 ---
	const erf = (x) => {
		const sign = x < 0 ? -1 : 1;
		x = Math.abs(x);
		const t = 1.0 / (1.0 + 0.3275911 * x);
		const y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
		return sign * y;
	};

	const pdf = (x) => {
		const invSqrt2Pi = 0.3989422804014327;
		return invSqrt2Pi * Math.exp(-0.5 * x * x);
	};

	const cdf = (x) => 0.5 * (1 + erf(x * 0.7071067811865475));

	// --- 参数处理器 ---
	const parameterProcessor = (() => {
		const paramKeys = Object.keys(searchSpace);
		const specs = paramKeys.map(key => {
			const def = searchSpace[key];
			if (Array.isArray(def)) return { key, type: 'cont', range: def };
			return { key, ...def };
		});

		const normalizedDim = specs.reduce((sum, s) => {
			if (s.type === 'cat') return sum + s.values.length;
			return sum + 1;
		}, 0);

		const rangeInfo = specs.map(spec => {
			if (spec.type === 'int' || spec.type === 'cont') {
				const [min, max] = spec.range;
				return { min, max, range: max - min, invRange: 1 / (max - min) };
			}
			return null;
		});

		const normalize = (params) => {
			const normalized = new Array(normalizedDim);
			let idx = 0;
			for (let i = 0; i < specs.length; i++) {
				const spec = specs[i];
				const value = params[spec.key];
				if (spec.type === 'int' || spec.type === 'cont') {
					const info = rangeInfo[i];
					normalized[idx++] = (Math.max(info.min, Math.min(info.max, value)) - info.min) * info.invRange;
				} else if (spec.type === 'cat') {
					const catIndex = spec.values.indexOf(value);
					for (let j = 0; j < spec.values.length; j++) {
						normalized[idx + j] = j === catIndex ? 1 : 0;
					}
					idx += spec.values.length;
				}
			}
			return normalized;
		};

		const denormalize = (normalized) => {
			const params = {};
			let idx = 0;
			for (let i = 0; i < specs.length; i++) {
				const spec = specs[i];
				if (spec.type === 'int' || spec.type === 'cont') {
					const info = rangeInfo[i];
					let val = normalized[idx++] * info.range + info.min;
					if (spec.type === 'int') {
						val = Math.round(val);
					}
					params[spec.key] = val;
				} else if (spec.type === 'cat') {
					let maxVal = -Infinity;
					let bestIndex = 0;
					for (let j = 0; j < spec.values.length; j++) {
						if (normalized[idx + j] > maxVal) {
							maxVal = normalized[idx + j];
							bestIndex = j;
						}
					}
					params[spec.key] = spec.values[bestIndex];
					idx += spec.values.length;
				}
			}
			return params;
		};

		return { normalize, denormalize, specs, normalizedDim };
	})();

	// --- 拉丁超立方采样 ---
	const generateLHSSamples = (numSamples) => {
		const dims = parameterProcessor.specs.length;
		const samples = [];
		
		const grid = Array(dims).fill(0).map(() => {
			const divisions = Array(numSamples).fill(0).map((_, i) => (i + rng()) / numSamples);
			for (let i = numSamples - 1; i > 0; i--) {
				const j = Math.floor(rng() * (i + 1));
				[divisions[i], divisions[j]] = [divisions[j], divisions[i]];
			}
			return divisions;
		});

		for (let i = 0; i < numSamples; i++) {
			const tempParams = {};
			for (let j = 0; j < dims; j++) {
				const spec = parameterProcessor.specs[j];
				const gridVal = grid[j][i];
				
				if (spec.type === 'int' || spec.type === 'cont') {
					const [min, max] = spec.range;
					let val = gridVal * (max - min) + min;
					if (spec.type === 'int') {
						val = Math.round(val);
					}
					tempParams[spec.key] = val;
				} else if (spec.type === 'cat') {
					const catIndex = Math.floor(gridVal * spec.values.length);
					tempParams[spec.key] = spec.values[catIndex];
				}
			}
			samples.push(tempParams);
		}
		return samples;
	};

	// --- 高斯过程核心 ---
	const kernel = (x1, x2, lengthScales) => {
		let sum = 0;
		for (let i = 0; i < x1.length; i++) {
			const diff = x1[i] - x2[i];
			const scale = Math.max(lengthScales[i], 1e-6);
			sum += (diff * diff) / (scale * scale);
		}
		return Math.exp(-0.5 * sum);
	};

	const computeKernelMatrix = (observations, lengthScales, noiseLevel) => {
		const n = observations.length;
		const K = Array(n);
		for (let i = 0; i < n; i++) {
			K[i] = Array(n);
			const obsI = observations[i].normalized;
			for (let j = 0; j <= i; j++) {
				const value = kernel(obsI, observations[j].normalized, lengthScales);
				K[i][j] = value;
				K[j][i] = value;
			}
			K[i][i] += noiseLevel + 1e-8;
		}
		return K;
	};

	const choleskyDecomposition = (A) => {
		const n = A.length;
		const L = Array(n);
		for (let i = 0; i < n; i++) {
			L[i] = Array(n).fill(0);
		}

		for (let i = 0; i < n; i++) {
			for (let j = 0; j <= i; j++) {
				let sum = 0;
				for (let k = 0; k < j; k++) {
					sum += L[i][k] * L[j][k];
				}
				if (i === j) {
					const diagValue = A[i][i] - sum;
					if (diagValue < 1e-10) {
						L[i][j] = Math.sqrt(diagValue + 1e-6);
					} else {
						L[i][j] = Math.sqrt(diagValue);
					}
				} else {
					L[i][j] = (A[i][j] - sum) / L[j][j];
				}
			}
		}
		return L;
	};

	const solveTriangular = (L, b, lower = true) => {
		const n = L.length;
		const x = Array(n);
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

	// --- 自适应超参数优化器 ---
	const hyperparameterOptimizer = (observations, iteration, totalIterations) => {
		const y = observations.map(obs => obs.loss);
		const n = observations.length;
		const defaultNoise = 1e-6;
		
		if (n < 3) {
			return { 
				lengthScales: Array(parameterProcessor.normalizedDim).fill(1.0), 
				noiseLevel: defaultNoise 
			};
		}

		// 动态调整优化强度：早期更激进，后期更保守
		const progress = iteration / totalIterations;
		const optimizationIterations = progress < 0.3 ? 6 : progress < 0.7 ? 4 : 2;
		
		const scaleCandidates = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0];
		let bestLengthScales = Array(parameterProcessor.normalizedDim).fill(1.0);
		let bestLogLikelihood = -Infinity;

		let currentLengthScales = [...bestLengthScales];

		for (let iter = 0; iter < optimizationIterations; iter++) {
			let improved = false;
			
			for (let dim = 0; dim < parameterProcessor.normalizedDim; dim++) {
				let bestDimScale = currentLengthScales[dim];
				let bestDimLikelihood = -Infinity;

				for (const scale of scaleCandidates) {
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
							improved = true;
						}
					} catch (e) {
						continue;
					}
				}
				currentLengthScales[dim] = bestDimScale;
			}

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
			} catch(e) {
				// 继续优化
			}

			if (!improved && iter > 1) break;
		}

		return { lengthScales: bestLengthScales, noiseLevel: defaultNoise };
	};

	const gpPredict = (observations, xNew, hyperparams) => {
		const n = observations.length;
		let K, L;
		let noise = hyperparams.noiseLevel;
		
		for (let attempt = 0; attempt < 3; attempt++) {
			try {
				K = computeKernelMatrix(observations, hyperparams.lengthScales, noise);
				L = choleskyDecomposition(K);
				break;
			} catch (e) {
				noise = attempt === 0 ? 1e-5 : noise * 10;
				if (attempt === 2) {
					K = computeKernelMatrix(observations, hyperparams.lengthScales, 0.1);
					for (let i = 0; i < n; i++) K[i][i] += 0.1;
					L = choleskyDecomposition(K);
				}
			}
		}

		const kStar = Array(n);
		const y = observations.map(obs => obs.loss);
		for (let i = 0; i < n; i++) {
			kStar[i] = kernel(observations[i].normalized, xNew, hyperparams.lengthScales);
		}
		
		const alpha = solveTriangular(L, solveTriangular(L, y, true), false);
		let mean = 0;
		for (let i = 0; i < n; i++) {
			mean += kStar[i] * alpha[i];
		}
		
		const v = solveTriangular(L, kStar, true);
		let variance = 1.0;
		for (let i = 0; i < n; i++) {
			variance -= v[i] * v[i];
		}

		return { mean, variance: Math.max(variance, 1e-6) };
	};

	// --- 动态采集函数 ---
	const acquisitionFunctions = {
		EI: (mean, std, bestLoss, xi = 0.01, progress = 0) => {
			if (std < 1e-12) return 0;
			// 动态调整xi：后期增加探索
			const dynamicXi = xi * (1 + progress * 2);
			const improvement = bestLoss - mean - dynamicXi;
			if (improvement <= 0) return 0;
			const z = improvement / std;
			return improvement * cdf(z) + std * pdf(z);
		},
		UCB: (mean, std, bestLoss, kappa = 2.576, progress = 0) => {
			// 动态调整kappa：后期增加探索
			const dynamicKappa = kappa * (1 + progress);
			return -mean + dynamicKappa * std;
		},
		PI: (mean, std, bestLoss, xi = 0.01, progress = 0) => {
			if (std < 1e-12) return (bestLoss - mean - xi) > 0 ? 1 : 0;
			return cdf((bestLoss - mean - xi) / std);
		}
	};

	// --- 平衡的候选点查找器 ---
	const candidateFinder = (observations, bestLoss, numCandidates, hyperparams, progress) => {
		// 早期阶段：更多随机探索
		if (observations.length < 5 || progress < 0.2) {
			return Array(numCandidates).fill(0).map(() => 
				parameterProcessor.denormalize(
					Array(parameterProcessor.normalizedDim).fill(0).map(() => rng())
				)
			);
		}

		const acquisitionFunc = acquisitionFunctions[acquisitionFunction] || acquisitionFunctions.EI;
		
		// 动态调整种群大小和演化步数
		const basePopulationSize = 40;
		const populationSize = Math.min(basePopulationSize + Math.floor(progress * 30), 70);
		const evolutionSteps = Math.max(15, 25 - Math.floor(progress * 10));

		// 1. 混合初始化策略
		let population = [];
		
		// 策略A：当前最佳点周围搜索 (局部开发)
		if (observations.length > 0) {
			const bestObs = observations.reduce((best, curr) => 
				curr.loss < best.loss ? curr : best
			);
			const localSearchCount = Math.floor(populationSize * (0.3 + progress * 0.2));
			for (let i = 0; i < localSearchCount; i++) {
				const mutated = bestObs.normalized.map(val => {
					// 动态调整变异范围：后期更精细
					const mutationRange = 0.2 * (1 - progress * 0.5);
					let newVal = val + (rng() - 0.5) * mutationRange;
					return Math.max(0, Math.min(1, newVal));
				});
				const { mean, variance } = gpPredict(observations, mutated, hyperparams);
				const acqValue = acquisitionFunc(mean, Math.sqrt(variance), bestLoss, undefined, progress);
				population.push({ normalized: mutated, acqValue });
			}
		}

		// 策略B：全局随机探索
		const globalSearchCount = Math.floor(populationSize * (0.4 - progress * 0.2));
		for (let i = 0; i < globalSearchCount; i++) {
			const randomPoint = Array(parameterProcessor.normalizedDim).fill(0).map(() => rng());
			const { mean, variance } = gpPredict(observations, randomPoint, hyperparams);
			const acqValue = acquisitionFunc(mean, Math.sqrt(variance), bestLoss, undefined, progress);
			population.push({ normalized: randomPoint, acqValue });
		}

		// 策略C：基于采集函数的高分点周围搜索
		while (population.length < populationSize) {
			const randomPoint = Array(parameterProcessor.normalizedDim).fill(0).map(() => rng());
			const { mean, variance } = gpPredict(observations, randomPoint, hyperparams);
			const acqValue = acquisitionFunc(mean, Math.sqrt(variance), bestLoss, undefined, progress);
			population.push({ normalized: randomPoint, acqValue });
		}

		// 2. 改进的演化策略
		for (let step = 0; step < evolutionSteps; step++) {
			const mutationStrength = 0.15 * (1 - step / evolutionSteps) * (1 - progress * 0.3);
			
			const offspring = [];
			for (let i = 0; i < populationSize / 2; i++) {
				// 锦标赛选择
				const a = population[Math.floor(rng() * populationSize)];
				const b = population[Math.floor(rng() * populationSize)];
				const parent = a.acqValue > b.acqValue ? a : b;
				
				// 混合变异策略
				const childNormalized = parent.normalized.map(val => {
					// 50%概率使用高斯变异，50%使用均匀变异
					let mutated;
					if (rng() < 0.5) {
						// 高斯变异
						mutated = val + (rng() - 0.5) * mutationStrength;
					} else {
						// 均匀变异
						mutated = val + (rng() - 0.5) * mutationStrength * 2;
					}
					return Math.max(0, Math.min(1, mutated));
				});
				
				const { mean, variance } = gpPredict(observations, childNormalized, hyperparams);
				const acqValue = acquisitionFunc(mean, Math.sqrt(variance), bestLoss, undefined, progress);
				offspring.push({ normalized: childNormalized, acqValue });
			}

			// 环境选择：保留最优个体
			population = [...population, ...offspring]
				.sort((a, b) => b.acqValue - a.acqValue)
				.slice(0, populationSize);
		}

		// 3. 多样性选择
		const selected = [];
		const usedPoints = new Set();
		
		for (let i = 0; i < population.length && selected.length < numCandidates; i++) {
			const candidate = population[i];
			const pointKey = candidate.normalized.map(v => Math.round(v * 10)).join(',');
			
			if (!usedPoints.has(pointKey)) {
				selected.push(candidate);
				usedPoints.add(pointKey);
			}
		}

		// 如果多样性不足，补充随机点
		while (selected.length < numCandidates) {
			const randomPoint = Array(parameterProcessor.normalizedDim).fill(0).map(() => rng());
			selected.push({ normalized: randomPoint, acqValue: 0 });
		}

		return selected.map(p => parameterProcessor.denormalize(p.normalized));
	};

	// --- 智能观测点选择 ---
	const selectMostInformativeObservations = (observations, maxSize, progress) => {
		if (observations.length <= maxSize) return observations;
		
		const sorted = [...observations].sort((a, b) => a.loss - b.loss);
		
		// 动态调整策略：早期保留更多多样性，后期聚焦最佳区域
		const bestRatio = 0.3 + progress * 0.3; // 从30%增加到60%
		const bestCount = Math.min(Math.floor(maxSize * bestRatio), 15);
		const selected = sorted.slice(0, bestCount);
		
		// 多样性选择
		const remaining = sorted.slice(bestCount);
		while (selected.length < maxSize && remaining.length > 0) {
			if (progress < 0.5) {
				// 早期：基于空间多样性
				let bestDiversity = -1;
				let bestIndex = -1;
				
				for (let i = 0; i < remaining.length; i++) {
					let minDistance = Infinity;
					for (const sel of selected) {
						const dist = remaining[i].normalized.reduce((sum, val, idx) => {
							const diff = val - sel.normalized[idx];
							return sum + diff * diff;
						}, 0);
						minDistance = Math.min(minDistance, dist);
					}
					if (minDistance > bestDiversity) {
						bestDiversity = minDistance;
						bestIndex = i;
					}
				}
				
				if (bestIndex >= 0) {
					selected.push(remaining.splice(bestIndex, 1)[0]);
				} else {
					break;
				}
			} else {
				// 后期：基于不确定性（如果可用）或随机选择
				const randomIndex = Math.floor(rng() * remaining.length);
				selected.push(remaining.splice(randomIndex, 1)[0]);
			}
		}
		
		return selected;
	};

	// --- 主优化循环 ---
	let observations = [];
	let bestLoss = Infinity;
	let bestParams = null;

	// 初始化
	const initialSamples = Math.max(8, parallel * 2); // 稍微减少初始样本
	const initialParamsList = generateLHSSamples(initialSamples);
	
	if (debug) console.log(`Starting optimization with ${initialSamples} initial samples...`);
	
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
	
	let iteration = initialSamples;
	if (debug) console.log(`Initial best loss: ${bestLoss.toFixed(6)}`);

	// 主循环
	while (iteration < iterations) {
		const progress = iteration / iterations;
		const batchSize = Math.min(parallel, iterations - iteration);
		
		const hyperparams = hyperparameterOptimizer(observations, iteration, iterations);
		const candidates = candidateFinder(observations, bestLoss, batchSize, hyperparams, progress);
		
		const batchResults = [];
		for (let i = 0; i < batchSize; i++) {
			const params = candidates[i];
			const idx = iteration + i;
			const loss = await objective(params, idx);
			batchResults.push({
				params, 
				normalized: parameterProcessor.normalize(params), 
				loss
			});
		}
		
		for (const result of batchResults) {
			observations.push(result);
			if (result.loss < bestLoss) {
				bestLoss = result.loss;
				bestParams = result.params;
				if (debug) {
					console.log(`[${iteration}] New best: ${bestLoss.toFixed(6)}, Progress: ${(progress * 100).toFixed(1)}%`);
				}
			}
		}
		
		iteration += batchResults.length;

		if (observations.length > maxObservations) {
			observations = selectMostInformativeObservations(observations, maxObservations, progress);
		}
	}

	if (debug) {
		console.log(`Optimization completed: ${iteration} iterations, best loss: ${bestLoss.toFixed(6)}`);
		console.log('Best parameters:', bestParams);
	}

	return { 
		bestParams, 
		bestLoss, 
		totalIterations: iteration
	};
};

export default BayesianOptimizer;
