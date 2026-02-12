
export const GeneticOptimizer = async (cfg) => {
    const {
        searchSpace,
        objective,
        iterations = 150,
		seed = null,
        debug = false,
    } = cfg;

    const paramNames = Object.keys(searchSpace);
    const paramRanges = Object.values(searchSpace);
    const dimensions = paramNames.length;

    // 自适应种群配置
    const basePopulationSize = Math.max(20, Math.min(30, Math.floor(iterations / 4)));
    const populationSize = basePopulationSize;

    // 高质量随机数生成器
    const rng = (() => {
		let state = seed !== null ? seed : Math.random() * 1000000;
		return () => {
			state = (state * 9301 + 49297) % 233280;
			return state / 233280;
		};
	})();

    // 高斯变异函数
    const gaussianRandom = (mean = 0, sigma = 1) => {
        let u1 = rng();
        let u2 = rng();
        let z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
        return z0 * sigma + mean;
    };

    // 批量评估 - 优化性能
    const evaluatePopulation = async (population, startIndex) => {
        const promises = population.map((individual, idx) => {
            const params = {};
            for (let j = 0; j < dimensions; j++) {
                params[paramNames[j]] = individual[j];
            }
            return objective(params, startIndex + idx);
        });
        return await Promise.all(promises);
    };

    // 智能个体创建 - 多策略组合
    const createIndividual = (bestIndividual = null, strategy = 'balanced') => {
        const individual = new Array(dimensions);
        
        if (bestIndividual && strategy === 'exploit') {
            // 围绕最佳个体进行精细搜索
            for (let i = 0; i < dimensions; i++) {
                const [min, max] = paramRanges[i];
                const range = max - min;
                const perturbation = gaussianRandom(0, range * 0.05); // 5%范围的高斯扰动
                individual[i] = Math.max(min, Math.min(max, bestIndividual[i] + perturbation));
            }
        } else if (strategy === 'explore') {
            // 探索性搜索 - 在参数空间的不同区域
            for (let i = 0; i < dimensions; i++) {
                const [min, max] = paramRanges[i];
                // 选择不同的搜索区域
                const region = Math.floor(rng() * 5); // 5个不同区域
                const regionCenters = [0.1, 0.3, 0.5, 0.7, 0.9];
                const regionCenter = min + (max - min) * regionCenters[region];
                const regionRange = (max - min) * 0.1; // 10%范围
                individual[i] = regionCenter + gaussianRandom(0, regionRange * 0.5);
                individual[i] = Math.max(min, Math.min(max, individual[i]));
            }
        } else {
            // 平衡策略 - 混合方法
            for (let i = 0; i < dimensions; i++) {
                const [min, max] = paramRanges[i];
                if (bestIndividual && rng() < 0.5) {
                    // 50%概率围绕最佳个体
                    const perturbation = (rng() - 0.5) * (max - min) * 0.08;
                    individual[i] = Math.max(min, Math.min(max, bestIndividual[i] + perturbation));
                } else {
                    // 50%概率完全随机
                    individual[i] = min + (max - min) * rng();
                }
            }
        }
        
        return individual;
    };

    // 自适应突变策略
    const mutate = (individual, progress, fitnessRank) => {
        const mutated = [...individual];
        
        // 基于进度和适应度排名的自适应突变率
        const baseMutationRate = 0.25 * (1 - progress * 0.3); // 最高25%，逐渐降低
        const rankFactor = 1 - (fitnessRank / populationSize); // 排名越差，突变率越高
        const adaptiveMutationRate = baseMutationRate * (0.7 + 0.3 * rankFactor);
        
        for (let i = 0; i < dimensions; i++) {
            if (rng() < adaptiveMutationRate) {
                const [min, max] = paramRanges[i];
                const currentValue = individual[i];
                const range = max - min;
                
                // 多策略自适应突变
                let mutationSize;
                const strategy = rng();
                
                if (strategy < 0.6) {
                    // 60%概率：自适应高斯突变
                    const sigma = range * (0.03 + progress * 0.02); // 3-5%范围
                    mutationSize = gaussianRandom(0, sigma);
                } else if (strategy < 0.85) {
                    // 25%概率：均匀突变
                    mutationSize = (rng() - 0.5) * range * 0.08;
                } else {
                    // 15%概率：较大跳跃
                    mutationSize = (rng() - 0.5) * range * 0.15;
                }
                
                let newValue = currentValue + mutationSize;
                
                // 边界反射处理
                if (newValue < min) newValue = min + (min - newValue) * 0.3;
                if (newValue > max) newValue = max - (newValue - max) * 0.3;
                newValue = Math.max(min, Math.min(max, newValue));
                
                mutated[i] = newValue;
            }
        }
        
        return mutated;
    };

    // 改进的交叉策略
    const crossover = (parent1, parent2, progress) => {
        const child1 = new Array(dimensions);
        const child2 = new Array(dimensions);
        
        // 自适应交叉权重 - 早期探索，后期利用
        const explorationWeight = 0.4 + progress * 0.2; // 0.4-0.6
        const alpha = explorationWeight + (rng() - 0.5) * 0.2; // 加入随机性
        
        for (let i = 0; i < dimensions; i++) {
            // 算术交叉
            child1[i] = alpha * parent1[i] + (1 - alpha) * parent2[i];
            child2[i] = (1 - alpha) * parent1[i] + alpha * parent2[i];
            
            // 边界检查
            const [min, max] = paramRanges[i];
            child1[i] = Math.max(min, Math.min(max, child1[i]));
            child2[i] = Math.max(min, Math.min(max, child2[i]));
        }
        
        return [child1, child2];
    };

    // 改进的选择策略
    const tournamentSelect = (population, fitnesses, tournamentSize = 3) => {
        let bestIndex = Math.floor(rng() * population.length);
        let bestFitness = fitnesses[bestIndex];
        
        for (let i = 1; i < tournamentSize; i++) {
            const candidateIndex = Math.floor(rng() * population.length);
            if (fitnesses[candidateIndex] > bestFitness) {
                bestFitness = fitnesses[candidateIndex];
                bestIndex = candidateIndex;
            }
        }
        
        return population[bestIndex];
    };

    // 轮盘赌选择
    const rouletteSelect = (population, fitnesses) => {
        // 转换为正数并计算概率
        const minFitness = Math.min(...fitnesses);
        const adjustedFitnesses = fitnesses.map(f => f - minFitness + 1e-6);
        const totalFitness = adjustedFitnesses.reduce((sum, f) => sum + f, 0);
        
        let randomValue = rng() * totalFitness;
        for (let i = 0; i < population.length; i++) {
            randomValue -= adjustedFitnesses[i];
            if (randomValue <= 0) {
                return population[i];
            }
        }
        
        return population[population.length - 1];
    };

    // 多样化初始化种群
    let population = [];
    const initializationStrategies = ['explore', 'balanced', 'exploit'];
    
    for (let i = 0; i < populationSize; i++) {
        const strategy = initializationStrategies[Math.floor(rng() * initializationStrategies.length)];
        population.push(createIndividual(null, strategy));
    }

    let usedEvaluations = 0;
    let bestLoss = Infinity;
    let bestIndividual = null;
    let bestParams = null;
    
    // 改进的收敛监控
    let stagnationCount = 0;
    let lastBestLoss = Infinity;
    const lossHistory = [];
    let improvementHistory = [];
    const diversityHistory = [];

    if (debug) console.log('Starting enhanced optimization...');

    // 计算种群多样性
    const calculateDiversity = (pop) => {
        if (pop.length <= 1) return 0;
        let totalDistance = 0;
        const center = new Array(dimensions).fill(0);
        
        // 计算中心点
        for (const individual of pop) {
            for (let j = 0; j < dimensions; j++) {
                center[j] += individual[j];
            }
        }
        for (let j = 0; j < dimensions; j++) {
            center[j] /= pop.length;
        }
        
        // 计算平均距离
        for (const individual of pop) {
            let distance = 0;
            for (let j = 0; j < dimensions; j++) {
                distance += Math.pow(individual[j] - center[j], 2);
            }
            totalDistance += Math.sqrt(distance);
        }
        
        return totalDistance / pop.length;
    };

    // 主优化循环
    while (usedEvaluations < iterations) {
        const progress = usedEvaluations / iterations;
        
        // 评估当前种群
        const batchSize = Math.min(population.length, iterations - usedEvaluations);
        const currentBatch = population.slice(0, batchSize);
        const losses = await evaluatePopulation(currentBatch, usedEvaluations);
        usedEvaluations += batchSize;
        
        // 计算适应度
        const fitnesses = losses.map(loss => -loss);
        
        // 更新最佳解
        let improved = false;
        for (let i = 0; i < losses.length; i++) {
            if (losses[i] < bestLoss) {
                const improvement = bestLoss === Infinity ? 0 : bestLoss - losses[i];
                bestLoss = losses[i];
                bestIndividual = [...currentBatch[i]];
                
                bestParams = {};
                for (let j = 0; j < dimensions; j++) {
                    bestParams[paramNames[j]] = bestIndividual[j];
                }
                
                improved = true;
                improvementHistory.push(improvement);
                
                if (debug && improvement > 0) {
                    console.log(`[${usedEvaluations}] New best: ${bestLoss.toFixed(4)} (improvement: ${improvement.toFixed(4)})`);
                }
            }
        }
        
        lossHistory.push(bestLoss);
        diversityHistory.push(calculateDiversity(population));
        
        // 更新停滞计数
        if (improved) {
            stagnationCount = Math.max(0, stagnationCount - 2);
        } else {
            stagnationCount++;
        }
        
        lastBestLoss = bestLoss;
        
        // 智能多样性管理
        const currentDiversity = diversityHistory[diversityHistory.length - 1];
        const needsDiversity = stagnationCount > 3 || currentDiversity < 0.1;
        
        if (needsDiversity) {
            if (debug && stagnationCount > 3) console.log('Intelligent diversity injection...');
            
            // 计算个体排名
            const fitnessWithIndices = fitnesses.map((f, i) => [f, i]);
            fitnessWithIndices.sort((a, b) => a[0] - b[0]);
            
            // 替换较差的50%个体
            const replaceCount = Math.floor(populationSize * 0.5);
            const worstIndices = fitnessWithIndices.slice(0, replaceCount).map(([_, i]) => i);
            
            for (const idx of worstIndices) {
                if (rng() < 0.7) {
                    population[idx] = createIndividual(bestIndividual, 'explore');
                } else {
                    population[idx] = createIndividual();
                }
            }
            
            stagnationCount = Math.floor(stagnationCount * 0.7); // 部分重置
        }
        
        // 定期探索注入
        if (usedEvaluations % 15 === 0 && progress < 0.85) {
            const numToReplace = Math.max(1, Math.floor(populationSize * 0.15));
            for (let i = 0; i < numToReplace; i++) {
                const replaceIndex = Math.floor(rng() * populationSize);
                population[replaceIndex] = createIndividual(null, 'explore');
            }
        }
        
        // 提前终止条件
        if (usedEvaluations >= iterations) break;
        if (stagnationCount > 12 && progress > 0.7) {
            if (debug) console.log('Adaptive early termination');
            break;
        }
        
        // 创建新一代
        const newPopulation = [];
        
        // 精英保留 - 自适应数量
        const eliteCount = Math.min(4, Math.max(2, Math.floor(populationSize * 0.2)));
        const eliteIndices = [...fitnesses]
            .map((f, i) => [f, i])
            .sort((a, b) => b[0] - a[0])
            .slice(0, eliteCount)
            .map(([_, i]) => i);
        
        for (const idx of eliteIndices) {
            newPopulation.push([...population[idx]]);
        }
        
        // 计算个体排名用于自适应突变
        const rankedFitness = [...fitnesses].map((f, i) => [f, i]);
        rankedFitness.sort((a, b) => b[0] - a[0]);
        const rankMap = new Map();
        rankedFitness.forEach(([f, idx], rank) => {
            rankMap.set(idx, rank);
        });
        
        // 生成新个体
        while (newPopulation.length < populationSize) {
            // 自适应选择策略
            const parent1 = progress < 0.6 ? tournamentSelect(population, fitnesses) : rouletteSelect(population, fitnesses);
            const parent2 = progress < 0.6 ? tournamentSelect(population, fitnesses) : rouletteSelect(population, fitnesses);
            
            let child1, child2;
            
            // 自适应交叉概率
            const crossoverProb = 0.85 - progress * 0.1; // 85%降至75%
            if (rng() < crossoverProb) {
                [child1, child2] = crossover(parent1, parent2, progress);
            } else {
                child1 = [...parent1];
                child2 = [...parent2];
            }
            
            // 自适应突变
            const rank1 = rankMap.get(population.indexOf(parent1)) || 0;
            const rank2 = rankMap.get(population.indexOf(parent2)) || 0;
            const avgRank = (rank1 + rank2) / 2;
            
            child1 = mutate(child1, progress, avgRank);
            child2 = mutate(child2, progress, avgRank);
            
            newPopulation.push(child1);
            if (newPopulation.length < populationSize) {
                newPopulation.push(child2);
            }
        }
        
        population = newPopulation.slice(0, populationSize);
        
        if (debug && usedEvaluations % 10 === 0) {
            const avgLoss = losses.reduce((a, b) => a + b, 0) / losses.length;
            console.log(`[${usedEvaluations}/${iterations}] Best: ${bestLoss.toFixed(4)}, Avg: ${avgLoss.toFixed(4)}, Stagnation: ${stagnationCount}`);
        }
    }

    const result = {
        bestParams,
        bestLoss,
        totalEvaluations: usedEvaluations,
        converged: stagnationCount <= 8,
        lossHistory,
        improvementHistory,
        diversityHistory,
        finalStagnation: stagnationCount
    };

    if (debug) {
        console.log('\n=== Optimization Summary ===');
        console.log(`Final loss: ${bestLoss.toFixed(4)}`);
        console.log(`Total evaluations: ${usedEvaluations}`);
        console.log(`Stagnation count: ${stagnationCount}`);
        console.log(`Converged: ${result.converged}`);
        console.log('Best parameters:', bestParams);
        
        // 性能分析
        if (improvementHistory.length > 0) {
            const significantImprovements = improvementHistory.filter(imp => imp > 0.1);
            console.log(`Significant improvements: ${significantImprovements.length}`);
            
            if (significantImprovements.length > 0) {
                const avgImprovement = significantImprovements.reduce((sum, imp) => sum + imp, 0) / significantImprovements.length;
                console.log(`Average significant improvement: ${avgImprovement.toFixed(4)}`);
            }
        }
        
        const finalDiversity = diversityHistory[diversityHistory.length - 1] || 0;
        console.log(`Final diversity: ${finalDiversity.toFixed(4)}`);
    }

    return result;
};

export default GeneticOptimizer;
