import { GeneticOptimizer as Optimizer } from "./GeneticOptimizer_v2.js";

const createSeededRandom = (seed = 27) => () => {
	let t = seed += 0x6D2B79F5;
	t = Math.imul(t ^ t >>> 15, t | 1);
	t ^= t + Math.imul(t ^ t >>> 7, t | 61);
	return ((t ^ t >>> 14) >>> 0) / 4294967296;
};
const random = createSeededRandom(123);

const searchSpace = {
	x: [-10, 10],
	y: [0, 10],
	z: [0, 10],
	a: [-50, 50],
};

// 根据搜索范围, 随机生成一些假设最优参数
const optimalParamsList = [];
for(let i = 0; i < 10; i++) {
	const optimalParams = [];
	for(const key in searchSpace) {
		const [ min, max ] = searchSpace[key];
		const value = random() * (max - min) + min;
		optimalParams.push(value);
	}
	optimalParamsList.push(optimalParams);
}

// 重复运行测试
const bestResultList = [];
for(const optimalParams of optimalParamsList){
	// 使用示例
	const bestResult = await Optimizer({

		// 目标函数
		// 你可以在这里添加各种东西, 比如运行一些其他软件, 并对结果进行评估
		// 确保此函数的输出值越接近 0 表示输入的参数越合适即可
		objective: (params, idx) => {
			const { x, y, z, a } = params;

			// 设定最优参数是这些
			// const optimalParams = [-5, 10, 5, 25];

			// 输入参数距离设定最优参数越近时, loss 越接近 0
			const loss = Math.abs(x - optimalParams[0]) + Math.abs(y - optimalParams[1]) + Math.abs(z - optimalParams[2]) + Math.abs(a - optimalParams[3]);

			// console.log(`[${idx}] Loss: ${loss}`);

			return loss;
		},

		// 定义参数搜索空间
		searchSpace,

		// 定义迭代次数
		iterations: 50,

		// 定义随机数种子
		seed: 123,

		// 调试模式 (打印日志)
		debug: false,
	});

	console.log('最优损失:', bestResult.bestLoss);

	bestResultList.push(bestResult);
}

// 计算平均值和累计 loss
const bestLosses = bestResultList.map(result => result.bestLoss);

// 计算平均 loss
const averageLoss = bestLosses.reduce((sum, loss) => sum + loss, 0) / bestLosses.length;

// 计算累计 loss
const totalLoss = bestLosses.reduce((sum, loss) => sum + loss, 0);

// 计算最小 loss
const minLoss = Math.min(...bestLosses);

// 计算最大 loss
const maxLoss = Math.max(...bestLosses);

// 计算标准差
const lossVariance = bestLosses.reduce((sum, loss) => sum + Math.pow(loss - averageLoss, 2), 0) / bestLosses.length;
const standardDeviation = Math.sqrt(lossVariance);

console.log('\n=== 性能统计 ===');
console.log(`平均损失: ${averageLoss.toFixed(4)}`);
console.log(`累计损失: ${totalLoss.toFixed(4)}`);
console.log(`最小损失: ${minLoss.toFixed(4)}`);
console.log(`最大损失: ${maxLoss.toFixed(4)}`);
console.log(`标准差: ${standardDeviation.toFixed(4)}`);
