import { BayesianOptimizer } from "./BayesianOptimizer_v4.js";

// 使用示例
const bestResult = await BayesianOptimizer({

	// 目标函数
	// 你可以在这里添加各种东西, 比如运行一些其他软件, 并对结果进行评估
	// 确保此函数的输出值越接近 0 表示输入的参数越合适即可
	objective: (params, idx) => {
		const { x, y, z, a } = params;

		// 设定最优参数是这些
		const optimalParams = [-5, 10, 5, 25];

		// 输入参数距离设定最优参数越近时, loss 越接近 0
		const loss = Math.abs(x - optimalParams[0]) + Math.abs(y - optimalParams[1]) + Math.abs(z - optimalParams[2]) + Math.abs(a - optimalParams[3]);

		console.log(`[${idx}] Loss: ${loss}`);

		return loss;
	},

	// 定义参数搜索空间
	searchSpace: {
		x: [-10, 10],
		y: [0, 10],
		z: [0, 10],
		a: [-50, 50],
	},

	// 定义迭代次数
	iterations: 50,

	// 定义随机数种子
	seed: 123,

	// 调试模式 (打印日志)
	debug: false,
});

console.log(`Best result:`, bestResult);
