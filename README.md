# Hyperparameter Optimizer

一个使用起来尽可能简单的超参数优化器.

```js
import HyperparameterOptimizer from './HyperparameterOptimizer.js';

// 目标函数
const objective = (trial) => {
	// 设定参数及搜索空间
	const x = trial.suggest_float('x', -10, 10);
	return Math.pow(x - 2, 2);
};

// 运行测试
const study = new HyperparameterOptimizer();
await study.optimize(objective);

// 得到结果
console.log('best_params:', study.best_params);
console.log('best_loss:', study.best_loss);
```


### 完整功能示例

```js
import HyperparameterOptimizer from './HyperparameterOptimizer.js';

// 目标函数
const objective = async (trial) => {
	const x = trial.suggest_float('x', -10, 10);
	return Math.pow(x - 2, 2);
};

// 运行测试
const study = new HyperparameterOptimizer();
await study.optimize(objective, {
	n: 100,
	seed: 42,
});
console.log('best_params:', study.best_params);
console.log('best_loss:', study.best_loss);

// 保存状态数据
const cache = study.save();

// 恢复状态数据继续测试
const study_2 = new HyperparameterOptimizer();
await study_2.optimize(objective, {
	n: 100,
	cache: cache,
});
console.log('best_params:', study_2.best_params);
console.log('best_loss:', study_2.best_loss);
```
