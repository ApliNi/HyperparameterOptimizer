/**
 * 贝叶斯优化器 - 使用高斯过程回归和采集函数
 */
export const BayesianOptimizer = async (config) => {
    const {
        objective,
        searchSpace,
        iterations = 100,
        initPoints = 10,
        acquisitionFunction = 'ei', // ei, ucb, poi
        kappa = 2.576, // UCB参数
        xi = 0.01, // EI, POI参数
        seed = null,
        debug = false
    } = config;

    // 随机数生成器
    const random = (() => {
        if (seed === null) return Math.random;
        let seedValue = seed;
        return () => {
            seedValue = (seedValue * 9301 + 49297) % 233280;
            return seedValue / 233280;
        };
    })();

    const paramNames = Object.keys(searchSpace);
    const numParams = paramNames.length;
    let evaluationCount = 0;
    
    // 存储观测数据
    const X = []; // 参数点
    const y = []; // 目标函数值
    
    let bestParams = null;
    let bestLoss = Infinity;

    // 高斯过程回归
    class GaussianProcess {
        constructor() {
            this.X = [];
            this.y = [];
            this.theta = 1.0; // 长度尺度
            this.sigma = 0.1; // 噪声标准差
        }
        
        // 径向基函数核
        kernel(x1, x2) {
            let sum = 0;
            for (let i = 0; i < x1.length; i++) {
                sum += Math.pow(x1[i] - x2[i], 2);
            }
            return Math.exp(-sum / (2 * this.theta * this.theta));
        }
        
        fit(X, y) {
            this.X = X;
            this.y = y;
        }
        
        // 预测均值和方差
        predict(x) {
            if (this.X.length === 0) {
                return { mean: 0, variance: 1 };
            }
            
            // 计算核向量
            const k = this.X.map(xi => this.kernel(xi, x));
            
            // 计算均值
            let mean = 0;
            if (this.X.length > 0) {
                // 简单加权平均
                const weights = k.map(val => val / (k.reduce((a, b) => a + b, 0) || 1));
                mean = weights.reduce((sum, weight, i) => sum + weight * this.y[i], 0);
            }
            
            // 计算方差（简化版本）
            const variance = Math.max(0.1, 1 - Math.max(...k));
            
            return { mean, variance };
        }
    }

    // 将参数对象转换为数组
    function paramsToArray(params) {
        return paramNames.map(name => params[name]);
    }

    // 将数组转换为参数对象
    function arrayToParams(arr) {
        const params = {};
        paramNames.forEach((name, i) => {
            params[name] = arr[i];
        });
        return params;
    }

    // 在搜索空间内随机采样
    function randomSample() {
        const params = {};
        for (const name of paramNames) {
            const [min, max] = searchSpace[name];
            params[name] = min + random() * (max - min);
        }
        return params;
    }

    // 拉丁超立方采样 - 更好的初始点覆盖
    function latinHypercubeSampling(n) {
        const samples = [];
        const intervals = [];
        
        // 为每个维度创建分层
        for (let dim = 0; dim < numParams; dim++) {
            intervals[dim] = [];
            for (let i = 0; i < n; i++) {
                intervals[dim].push([i / n, (i + 1) / n]);
            }
            // 随机打乱
            for (let i = intervals[dim].length - 1; i > 0; i--) {
                const j = Math.floor(random() * (i + 1));
                [intervals[dim][i], intervals[dim][j]] = [intervals[dim][j], intervals[dim][i]];
            }
        }
        
        // 生成样本
        for (let i = 0; i < n; i++) {
            const params = {};
            for (let dim = 0; dim < numParams; dim++) {
                const name = paramNames[dim];
                const [min, max] = searchSpace[name];
                const [lower, upper] = intervals[dim][i];
                const value = lower + random() * (upper - lower);
                params[name] = min + value * (max - min);
            }
            samples.push(params);
        }
        
        return samples;
    }

    // 评估目标函数
    async function evaluate(params) {
        const loss = await objective(params, evaluationCount);
        evaluationCount++;
        
        if (loss < bestLoss) {
            bestLoss = loss;
            bestParams = { ...params };
        }
        
        return loss;
    }

    // 期望改进（Expected Improvement）采集函数
    function expectedImprovement(mean, variance, bestY, xi = 0.01) {
        const std = Math.sqrt(variance);
        if (std === 0) return 0;
        
        const improvement = bestY - mean - xi;
        const z = improvement / std;
        
        const pdf = (1 / Math.sqrt(2 * Math.PI)) * Math.exp(-0.5 * z * z);
        const cdf = 0.5 * (1 + erf(z / Math.sqrt(2)));
        
        return improvement * cdf + std * pdf;
    }

    // 上置信界（Upper Confidence Bound）采集函数
    function upperConfidenceBound(mean, variance, kappa) {
        return mean - kappa * Math.sqrt(variance);
    }

    // 改进概率（Probability of Improvement）采集函数
    function probabilityOfImprovement(mean, variance, bestY, xi = 0.01) {
        const std = Math.sqrt(variance);
        if (std === 0) return 0;
        
        const z = (bestY - mean - xi) / std;
        return 0.5 * (1 + erf(z / Math.sqrt(2)));
    }

    // 误差函数
    function erf(x) {
        // 误差函数近似
        const a1 =  0.254829592;
        const a2 = -0.284496736;
        const a3 =  1.421413741;
        const a4 = -1.453152027;
        const a5 =  1.061405429;
        const p  =  0.3275911;

        const sign = (x < 0) ? -1 : 1;
        x = Math.abs(x);

        const t = 1.0 / (1.0 + p * x);
        const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);

        return sign * y;
    }

    // 选择下一个评估点
    function selectNextPoint(gp, bestY) {
        let bestAcquisition = -Infinity;
        let bestCandidate = null;
        
        // 在搜索空间内采样候选点
        const numCandidates = 1000;
        
        for (let i = 0; i < numCandidates; i++) {
            const candidate = randomSample();
            const x = paramsToArray(candidate);
            const prediction = gp.predict(x);
            
            let acquisitionValue;
            switch (acquisitionFunction) {
                case 'ucb':
                    acquisitionValue = -upperConfidenceBound(prediction.mean, prediction.variance, kappa);
                    break;
                case 'poi':
                    acquisitionValue = probabilityOfImprovement(prediction.mean, prediction.variance, bestY, xi);
                    break;
                case 'ei':
                default:
                    acquisitionValue = expectedImprovement(prediction.mean, prediction.variance, bestY, xi);
                    break;
            }
            
            if (acquisitionValue > bestAcquisition) {
                bestAcquisition = acquisitionValue;
                bestCandidate = candidate;
            }
        }
        
        return bestCandidate;
    }

    // 主优化循环
    console.log('=== 贝叶斯优化开始 ===');
    
    // 1. 初始采样
    const initialSamples = latinHypercubeSampling(initPoints);
    if (debug) console.log(`进行 ${initPoints} 个初始点的评估...`);
    
    for (const sample of initialSamples) {
        const loss = await evaluate(sample);
        X.push(paramsToArray(sample));
        y.push(loss);
        
        if (debug && evaluationCount % 5 === 0) {
            console.log(`[${evaluationCount}] Loss: ${loss.toFixed(4)}, Best: ${bestLoss.toFixed(4)}`);
        }
    }

    // 2. 贝叶斯优化循环
    if (debug) console.log(`开始贝叶斯优化循环...`);
    
    while (evaluationCount < iterations) {
        // 训练高斯过程
        const gp = new GaussianProcess();
        gp.fit(X, y);
        
        // 选择下一个点
        const nextPoint = selectNextPoint(gp, bestLoss);
        
        // 评估新点
        const loss = await evaluate(nextPoint);
        X.push(paramsToArray(nextPoint));
        y.push(loss);
        
        if (debug && evaluationCount % 10 === 0) {
            console.log(`[${evaluationCount}] Loss: ${loss.toFixed(4)}, Best: ${bestLoss.toFixed(4)}`);
        }
        
        // 早停检查
        if (evaluationCount > initPoints + 10) {
            const recentImprovements = y.slice(-10);
            const avgImprovement = recentImprovements.reduce((a, b) => a + b, 0) / recentImprovements.length;
            if (Math.abs(avgImprovement - bestLoss) < 1e-6) {
                if (debug) console.log('早停: 近期改进很小');
                break;
            }
        }
    }

    console.log('=== 贝叶斯优化完成 ===');
    
    return {
        bestParams,
        bestLoss,
        evaluations: evaluationCount,
        history: {
            X: X.map(arr => arrayToParams(arr)),
            y: y
        }
    };
};
