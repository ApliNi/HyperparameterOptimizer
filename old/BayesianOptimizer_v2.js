/**
 * 贝叶斯优化器 - 用于超参数优化的贝叶斯优化实现
 * @param {Object} cfg 配置对象
 * @param {Object} cfg.searchSpace 搜索空间定义，格式为 {paramName: [min, max], ...}
 * @param {Function} cfg.objective 目标函数，接受参数对象和迭代索引，返回Promise<number>
 * @param {number} cfg.iterations 最大迭代次数，默认50
 * @param {number} cfg.parallel 并行评估数量，默认1
 * @param {number|null} cfg.seed 随机数种子，默认null（使用随机种子）
 * @param {boolean} cfg.debug 是否输出调试信息，默认false
 * @param {string} cfg.acquisitionFunction 采集函数类型，支持'EI'、'UCB'、'PI'，默认'EI'
 * @param {number} cfg.maxObservations 最大观测点数量，用于限制内存使用，默认100
 * @returns {Promise<Object>} 优化结果，包含bestParams和bestLoss
 */
export const BayesianOptimizer = async (cfg) => {
	const {
		searchSpace,      // 搜索空间定义 {paramName: [min, max], ...}
		objective,        // 目标函数 (params, iteration) => Promise<loss>
		iterations = 50,  // 最大迭代次数
		parallel = 1,     // 并行评估数量
		seed = null,      // 随机数种子
		debug = false,    // 调试模式
		acquisitionFunction = 'EI', // 采集函数类型: 'EI', 'UCB', 'PI'
		maxObservations = 100       // 最大观测点数量限制
	} = cfg;

	// 随机数生成器
	const rng = (() => {
		let state = seed !== null ? seed : Math.random() * 1000000;
		return () => {
			state = (state * 9301 + 49297) % 233280;
			return state / 233280;
		};
	})();

	// 误差函数
	const erf = (x) => {
		const a1 = 0.254829592;
		const a2 = -0.284496736;
		const a3 = 1.421413741;
		const a4 = -1.453152027;
		const a5 = 1.061405429;
		const p = 0.3275911;

		const sign = x < 0 ? -1 : 1;
		x = Math.abs(x);

		const t = 1.0 / (1.0 + p * x);
		const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);

		return sign * y;
	};

	// 概率分布函数
	const pdf = (x) => Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
	const cdf = (x) => 0.5 * (1 + erf(x / Math.SQRT2));

	const normalizeParams = (params) => {
		const normalized = {};
		for (const [key, range] of Object.entries(searchSpace)) {
			// 避免边界数值问题
			const value = Math.max(range[0], Math.min(range[1], params[key]));
			normalized[key] = (value - range[0]) / (range[1] - range[0]);
			// 避免正好在0或1上
			normalized[key] = Math.min(0.999, Math.max(0.001, normalized[key]));
		}
		return normalized;
	};

	const denormalizeParams = (normalized) => {
		const params = {};
		for (const [key, range] of Object.entries(searchSpace)) {
			params[key] = normalized[key] * (range[1] - range[0]) + range[0];
		}
		return params;
	};

	// 拉丁超立方采样初始化
	const generateLHSSamples = (numSamples) => {
		const samples = [];
		const dims = Object.keys(searchSpace).length;
		
		// 生成LHS网格
		const grid = Array(dims).fill().map(() => {
			const divisions = Array(numSamples).fill().map((_, i) => (i + 0.5) / numSamples);
			// 随机打乱
			for (let i = numSamples - 1; i > 0; i--) {
				const j = Math.floor(rng() * (i + 1));
				[divisions[i], divisions[j]] = [divisions[j], divisions[i]];
			}
			return divisions;
		});
		
		// 生成样本
		for (let i = 0; i < numSamples; i++) {
			const normalized = {};
			let j = 0;
			for (const key of Object.keys(searchSpace)) {
				normalized[key] = grid[j][i];
				j++;
			}
			samples.push(denormalizeParams(normalized));
		}
		
		return samples;
	};

	// 生成随机参数（改进边界处理）
	const generateRandomParams = () => {
		const normalized = {};
		for (const key of Object.keys(searchSpace)) {
			normalized[key] = Math.min(0.999, Math.max(0.001, rng()));
		}
		return denormalizeParams(normalized);
	};

	// 核函数 (RBF) 带多个长度尺度
	const kernel = (x1, x2, lengthScales) => {
		let sum = 0;
		let i = 0;
		for (const key of Object.keys(searchSpace)) {
			const diff = x1[key] - x2[key];
			const scale = Array.isArray(lengthScales) ? lengthScales[i] : lengthScales;
			sum += (diff * diff) / (scale * scale);
			i++;
		}
		return Math.exp(-0.5 * sum);
	};

	// 计算核矩阵
	const computeKernelMatrix = (observations, lengthScales, noiseLevel = 1e-6) => {
		const n = observations.length;
		const K = Array(n).fill().map(() => Array(n).fill(0));
		
		for (let i = 0; i < n; i++) {
			for (let j = 0; j < n; j++) {
				K[i][j] = kernel(observations[i].normalized, observations[j].normalized, lengthScales);
				if (i === j) {
					K[i][j] += noiseLevel; // 噪声水平可调
				}
			}
		}
		return K;
	};

	// Cholesky 分解
	const choleskyDecomposition = (A) => {
		const n = A.length;
		const L = Array(n).fill().map(() => Array(n).fill(0));
		
		for (let i = 0; i < n; i++) {
			for (let j = 0; j <= i; j++) {
				let sum = 0;
				for (let k = 0; k < j; k++) {
					sum += L[i][k] * L[j][k];
				}
				
				if (i === j) {
					L[i][j] = Math.sqrt(Math.max(A[i][i] - sum, 1e-10));
				} else {
					L[i][j] = (A[i][j] - sum) / L[j][j];
				}
			}
		}
		return L;
	};

	// 解三角方程组
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

	// 边际似然优化超参数
	const optimizeHyperparameters = (observations) => {
		if (observations.length < 3) {
			// 初始阶段使用默认值
			const defaultLengthScale = 0.5;
			const defaultNoise = 1e-6;
			return { lengthScales: defaultLengthScale, noiseLevel: defaultNoise };
		}

		// 简单的网格搜索优化长度尺度
		const lengthScaleCandidates = [0.1, 0.2, 0.5, 1.0, 2.0];
		let bestLogLikelihood = -Infinity;
		let bestLengthScale = 0.5;
		let bestNoiseLevel = 1e-6;

		for (const lengthScale of lengthScaleCandidates) {
			try {
				const K = computeKernelMatrix(observations, lengthScale, 1e-6);
				const L = choleskyDecomposition(K);
				
				// 计算对数边际似然
				const y = observations.map(obs => obs.loss);
				const alpha = solveTriangular(L, solveTriangular(L, y, true), false);
				
				// log|K| = 2 * sum(log(diag(L)))
				let logDet = 0;
				for (let i = 0; i < L.length; i++) {
					logDet += Math.log(L[i][i]);
				}
				logDet *= 2;
				
				// 数据拟合项: y^T K^{-1} y = y^T alpha
				let dataFit = 0;
				for (let i = 0; i < y.length; i++) {
					dataFit += y[i] * alpha[i];
				}
				
				const logLikelihood = -0.5 * dataFit - logDet - 0.5 * y.length * Math.log(2 * Math.PI);
				
				if (logLikelihood > bestLogLikelihood) {
					bestLogLikelihood = logLikelihood;
					bestLengthScale = lengthScale;
				}
			} catch (e) {
				// 如果矩阵分解失败，跳过这个候选值
				continue;
			}
		}

		return { lengthScales: bestLengthScale, noiseLevel: bestNoiseLevel };
	};

	// 高斯过程预测
	const gpPredict = (observations, xNew, hyperparams) => {
		const n = observations.length;
		const K = computeKernelMatrix(observations, hyperparams.lengthScales, hyperparams.noiseLevel);
		const L = choleskyDecomposition(K);
		
		const kStar = observations.map(obs => 
			kernel(obs.normalized, xNew, hyperparams.lengthScales)
		);
		
		const y = observations.map(obs => obs.loss);
		const alpha = solveTriangular(L, solveTriangular(L, y, true), false);
		
		// 均值预测
		let mean = 0;
		for (let i = 0; i < n; i++) {
			mean += alpha[i] * kStar[i];
		}
		
		// 方差预测
		const v = solveTriangular(L, kStar, true);
		let variance = kernel(xNew, xNew, hyperparams.lengthScales);
		for (let i = 0; i < n; i++) {
			variance -= v[i] * v[i];
		}
		
		return { mean, variance: Math.max(variance, 0) };
	};

	// 多种采集函数
	const acquisitionFunctions = {
		EI: (mean, std, bestLoss, xi = 0.01) => {
			if (std === 0) return 0;
			const gamma = (bestLoss - mean - xi) / std;
			return std * (gamma * cdf(gamma) + pdf(gamma));
		},
		UCB: (mean, std, bestLoss, kappa = 2.576) => {
			return mean + kappa * std;
		},
		PI: (mean, std, bestLoss, xi = 0.01) => {
			if (std === 0) return 0;
			const gamma = (bestLoss - mean - xi) / std;
			return cdf(gamma);
		}
	};

	// 选择最有信息的观测点（用于限制内存）
	const selectMostInformativeObservations = (observations, maxSize) => {
		if (observations.length <= maxSize) return observations;
		
		// 简单策略：保留最好的点和一些随机点
		const sorted = [...observations].sort((a, b) => a.loss - b.loss);
		const bestCount = Math.floor(maxSize * 0.3);
		const randomCount = maxSize - bestCount;
		
		const selected = sorted.slice(0, bestCount);
		
		// 从剩余点中随机选择
		const remaining = sorted.slice(bestCount);
		for (let i = 0; i < randomCount && remaining.length > 0; i++) {
			const randomIndex = Math.floor(rng() * remaining.length);
			selected.push(remaining[randomIndex]);
			remaining.splice(randomIndex, 1);
		}
		
		return selected;
	};

	// 寻找下一个候选点
	const findNextCandidates = (observations, bestLoss, numCandidates, hyperparams) => {
		if (observations.length < 5) {
			// 初始阶段使用随机搜索
			return Array(numCandidates).fill().map(() => generateRandomParams());
		}

		const candidates = [];
		const acquisitionFunc = acquisitionFunctions[acquisitionFunction] || acquisitionFunctions.EI;
		
		for (let i = 0; i < numCandidates * 20; i++) {
			const candidateNormalized = {};
			for (const key of Object.keys(searchSpace)) {
				candidateNormalized[key] = Math.min(0.999, Math.max(0.001, rng()));
			}
			
			const prediction = gpPredict(observations, candidateNormalized, hyperparams);
			const std = Math.sqrt(prediction.variance);
			const acqValue = acquisitionFunc(prediction.mean, std, bestLoss);
			
			candidates.push({
				params: denormalizeParams(candidateNormalized),
				normalized: candidateNormalized,
				acqValue
			});
		}
		
		// 按采集函数值排序并选择最好的
		return candidates
			.sort((a, b) => b.acqValue - a.acqValue)
			.slice(0, numCandidates)
			.map(candidate => candidate.params);
	};

	// 主优化循环
	let observations = [];
	let bestLoss = Infinity;
	let bestParams = null;
	let iteration = 0;

	// 使用LHS进行初始采样
	const initialSamples = Math.max(5, parallel * 2);
	const initialParams = generateLHSSamples(initialSamples);
	
	for (let i = 0; i < initialSamples; i++) {
		const params = initialParams[i];
		const normalized = normalizeParams(params);
		const loss = await objective(params, i);
		
		observations.push({ params, normalized, loss });
		
		if (loss < bestLoss) {
			bestLoss = loss;
			bestParams = params;
		}
		
		if (debug) {
			console.log(`[Initial ${i}] Loss: ${loss}, Best: ${bestLoss}`);
		}
	}

	iteration = initialSamples;

	// 贝叶斯优化主循环
	while (iteration < iterations) {
		// 优化超参数
		const hyperparams = optimizeHyperparameters(observations);
		
		// 寻找候选点
		const candidates = findNextCandidates(observations, bestLoss, parallel, hyperparams);
		const promises = [];
		
		for (let i = 0; i < candidates.length && iteration < iterations; i++) {
			const idx = iteration++;
			promises.push(
				(async () => {
					const params = candidates[i];
					const normalized = normalizeParams(params);
					const loss = await objective(params, idx);
					
					observations.push({ params, normalized, loss });
					
					if (loss < bestLoss) {
						bestLoss = loss;
						bestParams = params;
						if (debug) {
							console.log(`[${idx}] New best: ${bestLoss}`);
						}
					}
					
					if (debug) {
						console.log(`[${idx}] Loss: ${loss}, Best: ${bestLoss}`);
					}
					
					return { params, loss };
				})()
			);
		}
		
		await Promise.all(promises);
		
		// 限制观测点数量防止内存爆炸
		if (observations.length > maxObservations) {
			observations = selectMostInformativeObservations(observations, maxObservations);
			if (debug) {
				console.log(`Reduced observations from ${observations.length + maxObservations} to ${maxObservations}`);
			}
		}
	}

	return {
		bestParams,
		bestLoss,
		totalIterations: iteration
	};
};

export default BayesianOptimizer;
