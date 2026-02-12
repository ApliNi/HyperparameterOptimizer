
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

const objective = [ objective_1, objective_2, objective_3, objective_4 ];

let best_value = 0;

for(const obj of objective){
	const study = new HyperparameterOptimizer();
	await study.optimize(obj, {
		n: 100,
		seed: 42,
	});

	const cache = study.save();

	const study_2 = new HyperparameterOptimizer();
	await study_2.optimize(obj, {
		n: 100,
		cache: cache,
	});
	console.log('best_loss:', study_2.best_loss);

	best_value += study_2.best_loss;
}

console.log(best_value);

