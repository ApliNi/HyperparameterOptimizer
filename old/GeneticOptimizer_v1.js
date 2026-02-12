/**
 * 遗传算法优化器 - 用于超参数优化的遗传算法实现
 * @param {Object} cfg 配置对象
 * @param {Object} cfg.searchSpace 搜索空间定义，格式为 {paramName: [min, max], ...}
 * @param {Function} cfg.objective 目标函数，接受参数对象和迭代索引，返回Promise<number>
 * @param {number} cfg.iterations 最大迭代次数，默认50
 * @param {number} cfg.populationSize 种群大小，默认20
 * @param {number|null} cfg.seed 随机数种子，默认null（使用随机种子）
 * @param {boolean} cfg.debug 是否输出调试信息，默认false
 * @returns {Promise<Object>} 优化结果，包含bestParams和bestLoss
 */
export const GeneticOptimizer = async (cfg) => {
    const {
        searchSpace,        // 搜索空间定义 {paramName: [min, max], ...}
        objective,          // 目标函数 (params, iteration) => Promise<loss>
        iterations = 50,    // 最大迭代次数
        populationSize = 20, // 种群大小
        seed = null,        // 随机数种子
        debug = false,      // 调试模式
    } = cfg;

    // 随机数生成器
    const rng = (() => {
        let state = seed !== null ? seed : Math.random() * 1000000;
        return () => {
            state = (state * 9301 + 49297) % 233280;
            return state / 233280;
        };
    })();

    // 创建初始种群
    const createRandomPhenotype = () => {
        const phenotype = {};
        let index = 0;
        for (const [key, range] of Object.entries(searchSpace)) {
            phenotype[index] = rng() * (range[1] - range[0]) + range[0];
            index++;
        }
        return phenotype;
    };

    // 将表型转换为参数对象
    const phenotypeToParams = (phenotype) => {
        const params = {};
        let index = 0;
        for (const key of Object.keys(searchSpace)) {
            params[key] = phenotype[index];
            index++;
        }
        return params;
    };

    // 将参数对象转换为表型
    const paramsToPhenotype = (params) => {
        const phenotype = {};
        let index = 0;
        for (const key of Object.keys(searchSpace)) {
            phenotype[index] = params[key];
            index++;
        }
        return phenotype;
    };

    // 适应度函数（最小化损失，所以用负值）
    const fitnessFunction = (phenotype) => {
        const params = phenotypeToParams(phenotype);
        // 这里不能直接调用objective，因为它是异步的
        // 我们将在主循环中处理
        return 0; // 占位符
    };

    // 突变函数
    const mutationFunction = (phenotype) => {
        const mutated = { ...phenotype };
        const keys = Object.keys(phenotype);
        
        // 随机选择1-3个基因进行突变
        const numMutations = Math.floor(rng() * 3) + 1;
        
        for (let i = 0; i < numMutations; i++) {
            const geneIndex = Math.floor(rng() * keys.length);
            const geneKey = keys[geneIndex];
            const range = searchSpace[Object.keys(searchSpace)[geneIndex]];
            
            // 随机突变
            const mutation = (rng() - 0.5) * (range[1] - range[0]) * 0.2;
            let newValue = mutated[geneKey] + mutation;
            
            // 确保值在范围内
            newValue = Math.max(range[0], Math.min(range[1], newValue));
            mutated[geneKey] = newValue;
        }
        
        return mutated;
    };

    // 交叉函数
    const crossoverFunction = (phenotypeA, phenotypeB) => {
        const child1 = {};
        const child2 = {};
        const keys = Object.keys(phenotypeA);
        
        for (let i = 0; i < keys.length; i++) {
            const key = keys[i];
            if (rng() < 0.5) {
                child1[key] = phenotypeA[key];
                child2[key] = phenotypeB[key];
            } else {
                child1[key] = phenotypeB[key];
                child2[key] = phenotypeA[key];
            }
        }
        
        return [child1, child2];
    };

    // 计算适应度（同步版本）
    const calculateFitness = async (phenotype, iteration) => {
        const params = phenotypeToParams(phenotype);
        const loss = await objective(params, iteration);
        return -loss; // 返回负值，因为我们要最小化损失
    };

    // 选择操作（锦标赛选择）
    const tournamentSelection = (population, fitnesses, tournamentSize = 3) => {
        const selected = [];
        
        for (let i = 0; i < population.length; i++) {
            let bestIndex = Math.floor(rng() * population.length);
            let bestFitness = fitnesses[bestIndex];
            
            for (let j = 1; j < tournamentSize; j++) {
                const candidateIndex = Math.floor(rng() * population.length);
                const candidateFitness = fitnesses[candidateIndex];
                
                if (candidateFitness > bestFitness) {
                    bestFitness = candidateFitness;
                    bestIndex = candidateIndex;
                }
            }
            
            selected.push(population[bestIndex]);
        }
        
        return selected;
    };

    // 主优化循环
    let population = Array(populationSize).fill().map(createRandomPhenotype);
    let bestLoss = Infinity;
    let bestParams = null;
    let bestPhenotype = null;

    // 初始评估
    const fitnesses = [];
    for (let i = 0; i < population.length; i++) {
        const loss = await objective(phenotypeToParams(population[i]), i);
        fitnesses.push(-loss);
        
        if (loss < bestLoss) {
            bestLoss = loss;
            bestParams = phenotypeToParams(population[i]);
            bestPhenotype = { ...population[i] };
        }
        
        if (debug) {
            console.log(`[${i}] Loss: ${loss}, Best: ${bestLoss}`);
        }
    }

    // 遗传算法主循环
    for (let generation = population.length; generation < iterations; generation++) {
        // 选择
        const selected = tournamentSelection(population, fitnesses);
        
        // 交叉和突变
        const newPopulation = [];
        
        // 保留最佳个体（精英主义）
        const bestIndex = fitnesses.indexOf(Math.max(...fitnesses));
        newPopulation.push({ ...population[bestIndex] });
        
        // 生成新个体
        while (newPopulation.length < populationSize) {
            const parent1 = selected[Math.floor(rng() * selected.length)];
            const parent2 = selected[Math.floor(rng() * selected.length)];
            
            let child1, child2;
            
            // 交叉
            if (rng() < 0.8) { // 交叉概率
                [child1, child2] = crossoverFunction(parent1, parent2);
            } else {
                child1 = { ...parent1 };
                child2 = { ...parent2 };
            }
            
            // 突变
            if (rng() < 0.1) { // 突变概率
                child1 = mutationFunction(child1);
            }
            if (rng() < 0.1) { // 突变概率
                child2 = mutationFunction(child2);
            }
            
            newPopulation.push(child1);
            if (newPopulation.length < populationSize) {
                newPopulation.push(child2);
            }
        }
        
        // 评估新种群
        population = newPopulation.slice(0, populationSize);
        fitnesses.length = 0;
        
        for (let i = 0; i < population.length; i++) {
            const loss = await objective(phenotypeToParams(population[i]), generation + i);
            fitnesses.push(-loss);
            
            if (loss < bestLoss) {
                bestLoss = loss;
                bestParams = phenotypeToParams(population[i]);
                bestPhenotype = { ...population[i] };
                
                if (debug) {
                    console.log(`[${generation + i}] New best: ${bestLoss}`);
                }
            }
            
            if (debug) {
                console.log(`[${generation + i}] Loss: ${loss}, Best: ${bestLoss}`);
            }
        }
    }

    return {
        bestParams,
        bestLoss,
        totalIterations: iterations
    };
};

export default GeneticOptimizer;
