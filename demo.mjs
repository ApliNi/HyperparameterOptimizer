
import HyperparameterOptimizer from './HyperparameterOptimizer.js';

const objective_1 = async (trial) => {
	const x = trial.suggest_float('x', -10, 10);
	return Math.pow(x - 2, 2);
};

const objective_2 = async (trial) => {
	const targets = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
	return targets.reduce((loss, target, i) => loss + Math.pow(trial.suggest_float(`x${i}`, -10, 20) - target, 2), 0);
};

const objective_3 = async (trial) => {
	const targets = [0, 5, 0, 5, 0, 5, 0, 5, 0, 5];
	return targets.reduce((loss, target, i) => loss + Math.pow(trial.suggest_float(`x${i}`, -10, 20) - target, 2), 0);
};

const objective_4 = async (trial) => {
	const targets = [0, 10, 0, 5, -10, 5, 0, 20, 0, 5];
	return targets.reduce((loss, target, i) => loss + Math.pow(trial.suggest_float(`x${i}`, -10, 20) - target, 2), 0);
};

// 选择目标函数
const objective = objective_3;

// 运行测试
const study = new HyperparameterOptimizer();
await study.optimize(objective, {
	n: 100,
	seed: 42,
});
console.log('best_params:', study.best_params);
console.log('best_loss:', study.best_loss);

// 保存数据
const cache = study.save();

// 恢复数据继续测试
const study_2 = new HyperparameterOptimizer();
await study_2.optimize(objective, {
	n: 100,
	cache: cache,
});
console.log('best_params:', study_2.best_params);
console.log('best_loss:', study_2.best_loss);
