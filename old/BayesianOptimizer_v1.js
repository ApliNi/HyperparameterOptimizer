

export const BayesianOptimizer = async (cfg) => {
	const {
		searchSpace,
		objective,
		iterations = 50,
		parallel = 1,
		seed = null,
		debug = false
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
		// 误差函数的近似计算
		const a1 =	0.254829592;
		const a2 =	-0.284496736;
		const a3 =	1.421413741;
		const a4 =	-1.453152027;
		const a5 =	1.061405429;
		const p  =	0.3275911;

		const sign = x < 0 ? -1 : 1;
		x = Math.abs(x);

		const t = 1.0 / (1.0 + p * x);
		const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);

		return sign * y;
	};
	
	const normalizeParams = (params) => {
		const normalized = {};
		for (const [key, range] of Object.entries(searchSpace)) {
			normalized[key] = (params[key] - range[0]) / (range[1] - range[0]);
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

	// 生成随机参数
	const generateRandomParams = () => {
		const normalized = {};
		for (const key of Object.keys(searchSpace)) {
			normalized[key] = rng();
		}
		return denormalizeParams(normalized);
	};

	// 核函数 (RBF)
	const kernel = (x1, x2, lengthScale = 1.0) => {
		let sum = 0;
		for (const key of Object.keys(searchSpace)) {
			const diff = x1[key] - x2[key];
			sum += diff * diff;
		}
		return Math.exp(-sum / (2 * lengthScale * lengthScale));
	};

	// 计算核矩阵
	const computeKernelMatrix = (observations, lengthScale) => {
		const n = observations.length;
		const K = Array(n).fill().map(() => Array(n).fill(0));
		
		for (let i = 0; i < n; i++) {
			for (let j = 0; j < n; j++) {
				K[i][j] = kernel(observations[i].normalized, observations[j].normalized, lengthScale);
				if (i === j) {
					K[i][j] += 1e-6; // 添加小的噪声确保矩阵正定
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

	// 高斯过程预测
	const gpPredict = (observations, xNew, lengthScale) => {
		const n = observations.length;
		const K = computeKernelMatrix(observations, lengthScale);
		const L = choleskyDecomposition(K);
		
		const kStar = observations.map(obs => 
			kernel(obs.normalized, xNew, lengthScale)
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
		let variance = kernel(xNew, xNew, lengthScale);
		for (let i = 0; i < n; i++) {
			variance -= v[i] * v[i];
		}
		
		return { mean, variance: Math.max(variance, 0) };
	};

	// 采集函数 (Expected Improvement)
	const expectedImprovement = (mean, std, bestLoss) => {
		if (std === 0) return 0;
		
		const gamma = (bestLoss - mean) / std;
		const pdf = (x) => Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
		const cdf = (x) => 0.5 * (1 + erf(x / Math.SQRT2));
		
		return std * (gamma * cdf(gamma) + pdf(gamma));
	};

	// 寻找下一个候选点
	const findNextCandidates = (observations, bestLoss, numCandidates) => {
		if (observations.length < 5) {
			// 初始阶段使用随机搜索
			return Array(numCandidates).fill().map(() => generateRandomParams());
		}

		const candidates = [];
		const lengthScale = 0.5; // 核长度尺度
		
		for (let i = 0; i < numCandidates * 10; i++) {
			const candidateNormalized = {};
			for (const key of Object.keys(searchSpace)) {
				candidateNormalized[key] = rng();
			}
			
			const prediction = gpPredict(observations, candidateNormalized, lengthScale);
			const std = Math.sqrt(prediction.variance);
			const ei = expectedImprovement(prediction.mean, std, bestLoss);
			
			candidates.push({
				params: denormalizeParams(candidateNormalized),
				normalized: candidateNormalized,
				ei
			});
		}
		
		// 按 EI 值排序并选择最好的
		return candidates
			.sort((a, b) => b.ei - a.ei)
			.slice(0, numCandidates)
			.map(candidate => candidate.params);
	};

	// 主优化循环
	const observations = [];
	let bestLoss = Infinity;
	let bestParams = null;
	let iteration = 0;

	// 初始随机采样
	const initialSamples = Math.max(5, parallel * 2);
	for (let i = 0; i < initialSamples; i++) {
		const params = generateRandomParams();
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
		const candidates = findNextCandidates(observations, bestLoss, parallel);
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
					}
					
					if (debug) {
						console.log(`[${idx}] Loss: ${loss}, Best: ${bestLoss}`);
					}
					
					return { params, loss };
				})()
			);
		}
		
		await Promise.all(promises);
	}

	return {
		bestParams,
		bestLoss
	};
};

export default BayesianOptimizer;
