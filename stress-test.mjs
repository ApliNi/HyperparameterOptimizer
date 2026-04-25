import { performance } from 'node:perf_hooks';
import HyperparameterOptimizer from './HyperparameterOptimizer.js';

const EPS = 1e-9;

function formatNumber(value) {
	if (!Number.isFinite(value)) return String(value);
	const abs = Math.abs(value);
	if (abs >= 1000 || (abs > 0 && abs < 0.001)) return value.toExponential(4);
	return value.toFixed(6);
};

function mean(values) {
	return values.reduce((sum, value) => sum + value, 0) / values.length;
};

function createDeterministicNoise(values, scale = 1) {
	let acc = 0;
	for (let i = 0; i < values.length; i++) {
		acc += Math.sin(values[i] * (i + 1) * 12.9898) * 43758.5453;
	}
	return (acc - Math.floor(acc) - 0.5) * scale;
};

function calcImprovementRate(loss, baselineLoss, targetLoss = 0) {
	const denominator = Math.max(EPS, baselineLoss - targetLoss);
	return Math.max(0, Math.min(1, (baselineLoss - loss) / denominator));
};

async function runOptimizer(objective, { n, seed, cache = null } = {}) {
	const optimizer = new HyperparameterOptimizer();
	const startedAt = performance.now();
	await optimizer.optimize(objective, { n, seed, cache });
	return {
		elapsedMs: performance.now() - startedAt,
		bestLoss: optimizer.best_loss,
		bestParams: optimizer.best_params,
		cache: optimizer.save(),
	};
};

async function runMicroBenchmark(name, repeats, objectiveFactory, optionsFactory) {
	const startedAt = performance.now();
	for (let i = 0; i < repeats; i++) {
		const optimizer = new HyperparameterOptimizer();
		await optimizer.optimize(objectiveFactory(), optionsFactory(i));
		optimizer.save();
	}
	const elapsedMs = performance.now() - startedAt;
	return {
		name,
		repeats,
		elapsedMs,
		avgMs: elapsedMs / repeats,
	};
};

async function evaluateIsolatedMicroBenchmarks() {
	console.log('\n=== 独立微基准 ===');
	const steadyPrimitiveCategorical = await runMicroBenchmark(
		'primitive_categorical_steady',
		240,
		() => async (trial) => {
			let loss = 0;
			for (let i = 0; i < 40; i++) {
				const mode = trial.suggest_categorical(`bench_mode_${i}`, ['cold', 'warm', 'hot', 'burst', 'idle', 'eco']);
				loss += mode === 'hot' ? 0 : 1;
			}
			return loss;
		},
		(index) => ({ n: 220, seed: 7000 + index }),
	);
	const dynamicPrimitiveCategorical = await runMicroBenchmark(
		'primitive_categorical_dynamic',
		180,
		() => async (trial) => {
			let loss = 0;
			for (let i = 0; i < 24; i++) {
				const toggle = (trial.trialParams[`bench_selector_${i}`] ?? 0.5) > 0.5;
				const choices = toggle ? ['cold', 'warm', 'hot'] : ['cold', 'warm', 'eco'];
				const mode = trial.suggest_categorical(`bench_selector_${i}`, choices);
				loss += mode === (toggle ? 'hot' : 'eco') ? 0 : 1;
			}
			return loss;
		},
		(index) => ({ n: 220, seed: 9000 + index }),
	);
	console.log(`${steadyPrimitiveCategorical.name}: total=${formatNumber(steadyPrimitiveCategorical.elapsedMs)} ms, avg=${formatNumber(steadyPrimitiveCategorical.avgMs)} ms/run`);
	console.log(`${dynamicPrimitiveCategorical.name}: total=${formatNumber(dynamicPrimitiveCategorical.elapsedMs)} ms, avg=${formatNumber(dynamicPrimitiveCategorical.avgMs)} ms/run`);
	return {
		steadyPrimitiveCategorical,
		dynamicPrimitiveCategorical,
	};
};

function printScenario(result) {
	console.log(`\n[${result.pass ? 'PASS' : 'WARN'}] ${result.name}`);
	console.log(`说明: ${result.description}`);
	console.log(`迭代: ${result.n}, 种子: ${result.seed}, 耗时: ${formatNumber(result.elapsedMs)} ms`);
	console.log(`损失: best=${formatNumber(result.bestLoss)}, baseline=${formatNumber(result.baselineLoss)}, target=${formatNumber(result.targetLoss)}`);
	console.log(`改善率: ${(result.improvementRate * 100).toFixed(2)}%, 门槛: ${(result.passRate * 100).toFixed(2)}%`);
	if (result.inspectMessage) console.log(`业务检查: ${result.inspectMessage}`);
	console.log(`最优参数: ${JSON.stringify(result.bestParams)}`);
};

async function evaluateScenario(config) {
	const run = await runOptimizer(config.objective, { n: config.n, seed: config.seed });
	const improvementRate = calcImprovementRate(run.bestLoss, config.baselineLoss, config.targetLoss ?? 0);
	const inspect = config.inspect ? config.inspect(run.bestParams, run.bestLoss) : { ok: true, message: '' };
	return {
		name: config.name,
		description: config.description,
		n: config.n,
		seed: config.seed,
		baselineLoss: config.baselineLoss,
		targetLoss: config.targetLoss ?? 0,
		bestLoss: run.bestLoss,
		bestParams: run.bestParams,
		elapsedMs: run.elapsedMs,
		improvementRate,
		passRate: config.passRate,
		pass: improvementRate >= config.passRate && inspect.ok,
		inspectMessage: inspect.message,
	};
};

async function evaluateResumeConsistency() {
	console.log('\n=== 断点续跑一致性 ===');
	const objective = async (trial) => {
		const x = trial.suggest_float('x', -10, 10);
		const y = trial.suggest_float('y', -6, 6);
		const z = trial.suggest_float('z', -3, 9);
		return Math.pow(x - 2.25, 2) + Math.pow(y + 1.5, 2) + Math.pow(z - 4.5, 2);
	};
	const staged = new HyperparameterOptimizer();
	let cache = null;
	for (const n of [35, 40, 45]) {
		await staged.optimize(objective, { n, seed: 2026, cache });
		cache = staged.save();
	}
	const direct = new HyperparameterOptimizer();
	await direct.optimize(objective, { n: 120, seed: 2026 });
	const diff = Math.abs(staged.best_loss - direct.best_loss);
	const sameParams = JSON.stringify(staged.best_params) === JSON.stringify(direct.best_params);
	console.log(`分段结果: ${formatNumber(staged.best_loss)}`);
	console.log(`整段结果: ${formatNumber(direct.best_loss)}`);
	console.log(`差异: ${formatNumber(diff)}`);
	console.log(`参数一致: ${sameParams}`);
	return diff < 1e-12 && sameParams;
};

async function evaluateDynamicSchemaResumeConsistency() {
	console.log('\n=== 动态 Schema 续跑一致性 ===');
	const objective = async (trial) => {
		const mode = trial.suggest_categorical('mode', ['A', 'B']);
		let loss = mode === 'B' ? 0 : 6;

		if (mode === 'A') {
			const sharedKnob = trial.suggest_int('shared_knob', 32, 256);
			loss += Math.pow((sharedKnob - 192) / 18, 2);
		} else {
			const sharedKnob = trial.suggest_float('shared_knob', 0.1, 1.0);
			const policy = trial.suggest_categorical('policy', ['p0', 'p1', 'p2']);
			loss += Math.pow((sharedKnob - 0.73) / 0.04, 2) + (policy === 'p2' ? 0 : 4);
		}

		return loss;
	};
	const staged = new HyperparameterOptimizer();
	let cache = null;
	for (const n of [40, 50, 60]) {
		await staged.optimize(objective, { n, seed: 42036, cache });
		cache = staged.save();
	}
	const direct = new HyperparameterOptimizer();
	await direct.optimize(objective, { n: 150, seed: 42036 });
	const diff = Math.abs(staged.best_loss - direct.best_loss);
	const sameParams = JSON.stringify(staged.best_params) === JSON.stringify(direct.best_params);
	console.log(`分段结果: ${formatNumber(staged.best_loss)}`);
	console.log(`整段结果: ${formatNumber(direct.best_loss)}`);
	console.log(`差异: ${formatNumber(diff)}`);
	console.log(`参数一致: ${sameParams}`);
	console.log(`params(staged): ${JSON.stringify(staged.best_params)}`);
	console.log(`params(direct): ${JSON.stringify(direct.best_params)}`);
	return diff < 1e-12 && sameParams;
};

async function evaluateNonPrimitiveCategoricalResumeConsistency() {
	console.log('\n=== 非 Primitive 分类续跑一致性 ===');
	const objective = async (trial) => {
		const pack = trial.suggest_categorical('pack', [
			{ name: 'small', cpu: 2, mem: 4 },
			{ name: 'medium', cpu: 4, mem: 8 },
			{ name: 'large', cpu: 8, mem: 16 },
		]);
		const route = trial.suggest_categorical('route', [
			['cold', 0.3],
			['warm', 0.6],
			['hot', 0.9],
		]);
		const ratio = trial.suggest_float('ratio', 0, 1);
		const packPenalty = pack.name === 'medium' ? 0 : 6;
		const routePenalty = route[0] === 'warm' ? 0 : 5;
		return packPenalty + routePenalty + Math.pow((ratio - 0.62) / 0.05, 2);
	};
	const staged = new HyperparameterOptimizer();
	let cache = null;
	for (const n of [45, 45, 45]) {
		await staged.optimize(objective, { n, seed: 52037, cache });
		cache = staged.save();
	}
	const direct = new HyperparameterOptimizer();
	await direct.optimize(objective, { n: 135, seed: 52037 });
	const diff = Math.abs(staged.best_loss - direct.best_loss);
	const sameParams = JSON.stringify(staged.best_params) === JSON.stringify(direct.best_params);
	console.log(`分段结果: ${formatNumber(staged.best_loss)}`);
	console.log(`整段结果: ${formatNumber(direct.best_loss)}`);
	console.log(`差异: ${formatNumber(diff)}`);
	console.log(`参数一致: ${sameParams}`);
	console.log(`params(staged): ${JSON.stringify(staged.best_params)}`);
	console.log(`params(direct): ${JSON.stringify(direct.best_params)}`);
	return diff < 1e-12 && sameParams;
};

async function evaluateNonPrimitiveCategoricalResetAfterResume() {
	console.log('\n=== 非 Primitive 分类 reset 续跑一致性 ===');
	const trainingObjective = async (trial) => {
		const pack = trial.suggest_categorical('reset_pack', [
			{ name: 'small', cpu: 2 },
			{ name: 'medium', cpu: 4 },
			{ name: 'large', cpu: 8 },
		]);
		return pack.name === 'medium' ? 0 : 10;
	};
	const freezeObjective = async (trial) => {
		trial.suggest_categorical('reset_pack', [
			{ name: 'small', cpu: 2 },
			{ name: 'medium', cpu: 4 },
			{ name: 'large', cpu: 8 },
		]);
		return 10;
	};
	const staged = new HyperparameterOptimizer();
	await staged.optimize(trainingObjective, { n: 80, seed: 62038 });
	const cache = staged.save();
	const resumed = new HyperparameterOptimizer();
	await resumed.optimize(freezeObjective, { n: 25, seed: 62038, cache });
	const internalIndex = resumed.state.params.reset_pack;
	const restoredChoice = resumed.state.meta.reset_pack?.choices?.[Math.round(internalIndex)];
	const ok = Math.round(internalIndex) === 1 && restoredChoice?.name === 'medium';
	console.log(`best(before save): ${JSON.stringify(staged.best_params)}`);
	console.log(`internal_index(after reset): ${formatNumber(internalIndex)}`);
	console.log(`restored_choice(after reset): ${JSON.stringify(restoredChoice)}`);
	return ok;
};

async function evaluateNonPrimitiveCategoricalMutationIsolation() {
	console.log('\n=== 非 Primitive 分类引用隔离一致性 ===');
	const optimizer = new HyperparameterOptimizer();
	const objective = async (trial) => {
		const pack = trial.suggest_categorical('alias_pack', [
			{ name: 'small', cpu: 2 },
			{ name: 'medium', cpu: 4 },
			{ name: 'large', cpu: 8 },
		]);
		const originalName = pack.name;
		const loss = originalName === 'medium' ? 0 : 10;
		pack.name = `${originalName}_mutated`;
		return loss;
	};
	await optimizer.optimize(objective, { n: 60, seed: 72038 });
	const bestName = optimizer.best_params?.alias_pack?.name;
	const metaName = optimizer.state.meta.alias_pack?.choices?.[1]?.name;
	const ok = bestName === 'medium' && metaName === 'medium';
	console.log(`best_name: ${JSON.stringify(bestName)}`);
	console.log(`meta_medium_name: ${JSON.stringify(metaName)}`);
	return ok;
};

async function evaluateCacheRestoreIsolation() {
	console.log('\n=== cache 恢复引用隔离一致性 ===');
	const baseObjective = async (trial) => {
		const x = trial.suggest_float('cache_iso_x', -5, 5);
		const gate = trial.suggest_bool('cache_iso_gate');
		return Math.pow(x - 1.25, 2) + (gate ? 0 : 3);
	};
	const resumedObjective = async (trial) => {
		const x = trial.suggest_float('cache_iso_x', -5, 5);
		const gate = trial.suggest_bool('cache_iso_gate');
		return Math.pow(x + 2.1, 2) + (gate ? 2 : 0);
	};
	const optimizer = new HyperparameterOptimizer();
	await optimizer.optimize(baseObjective, { n: 80, seed: 82038 });
	const cache = optimizer.save();
	const snapshot = structuredClone(cache);
	const resumed = new HyperparameterOptimizer();
	await resumed.optimize(resumedObjective, { n: 25, seed: 82039, cache });
	const sameCache = JSON.stringify(cache) === JSON.stringify(snapshot);
	console.log(`cache_unchanged: ${sameCache}`);
	console.log(`cache_best_before: ${JSON.stringify(snapshot.best_params)}`);
	console.log(`cache_best_after: ${JSON.stringify(cache.best_params)}`);
	return sameCache;
};

async function evaluateMetaRegistrationStability() {
	console.log('\n=== 元信息注册稳定性 ===');
	const optimizer = new HyperparameterOptimizer();
	const objective = async (trial) => {
		const ratio = trial.suggest_float('meta_ratio', -1, 1);
		const count = trial.suggest_int('meta_count', 1, 9);
		const gate = trial.suggest_bool('meta_gate');
		const mode = trial.suggest_categorical('meta_mode', ['cold', 'warm', 'hot']);
		return Math.pow(ratio - 0.25, 2) + Math.pow(count - 5, 2) + (gate ? 0 : 2) + (mode === 'warm' ? 0 : 1);
	};
	await optimizer.optimize(objective, { n: 80, seed: 91001 });
	const meta = optimizer.state.meta;
	const ok = meta.meta_ratio?.type === 'float'
		&& meta.meta_count?.type === 'int'
		&& meta.meta_gate?.type === 'bool'
		&& meta.meta_mode?.type === 'categorical'
		&& meta.meta_mode?.min === 0
		&& meta.meta_mode?.max === 2
		&& JSON.stringify(meta.meta_mode?.choices) === JSON.stringify(['cold', 'warm', 'hot']);
	console.log(`meta_ratio: ${JSON.stringify(meta.meta_ratio)}`);
	console.log(`meta_count: ${JSON.stringify(meta.meta_count)}`);
	console.log(`meta_gate: ${JSON.stringify(meta.meta_gate)}`);
	console.log(`meta_mode: ${JSON.stringify(meta.meta_mode)}`);
	return ok;
};

async function evaluateDeterminism() {
	console.log('\n=== 同种子复现性 ===');
	const objective = async (trial) => {
		const alpha = trial.suggest_float('alpha', -5, 5);
		const beta = trial.suggest_float('beta', -5, 5);
		return Math.pow(alpha - 1.2, 2) + Math.pow(beta + 2.4, 2);
	};
	const runA = await runOptimizer(objective, { n: 80, seed: 31415 });
	const runB = await runOptimizer(objective, { n: 80, seed: 31415 });
	console.log(`loss(A): ${formatNumber(runA.bestLoss)}`);
	console.log(`loss(B): ${formatNumber(runB.bestLoss)}`);
	console.log(`params(A): ${JSON.stringify(runA.bestParams)}`);
	console.log(`params(B): ${JSON.stringify(runB.bestParams)}`);
	return Math.abs(runA.bestLoss - runB.bestLoss) < 1e-12 && JSON.stringify(runA.bestParams) === JSON.stringify(runB.bestParams);
};

async function evaluateSeedSensitivity() {
	console.log('\n=== 多种子稳定性 ===');
	const objective = async (trial) => {
		const lr = trial.suggest_float('lr', 0.0001, 0.03);
		const dropout = trial.suggest_float('dropout', 0, 0.5);
		const hidden = trial.suggest_int('hidden', 32, 256);
		const activation = trial.suggest_categorical('activation', ['relu', 'gelu', 'swish']);
		return Math.pow((lr - 0.008) / 0.0025, 2) + Math.pow((dropout - 0.12) / 0.05, 2) + Math.pow((hidden - 160) / 24, 2) + (activation === 'gelu' ? 0 : activation === 'swish' ? 2 : 6);
	};
	const seeds = [1, 7, 21, 42, 99, 2026];
	const losses = [];
	for (const seed of seeds) {
		const run = await runOptimizer(objective, { n: 180, seed });
		losses.push(run.bestLoss);
		console.log(`seed=${seed}, best_loss=${formatNumber(run.bestLoss)}, params=${JSON.stringify(run.bestParams)}`);
	}
	const averageLoss = mean(losses);
	const worstLoss = Math.max(...losses);
	console.log(`平均损失: ${formatNumber(averageLoss)}`);
	console.log(`最差损失: ${formatNumber(worstLoss)}`);
	return worstLoss < 10 && averageLoss < 6;
};

async function evaluateExtendedSeedStability() {
	console.log('\n=== 扩展多种子基准 ===');
	const objective = async (trial) => {
		const lr = trial.suggest_float('lr_ext', 0.0001, 0.03);
		const dropout = trial.suggest_float('dropout_ext', 0, 0.5);
		const hidden = trial.suggest_int('hidden_ext', 32, 256);
		const activation = trial.suggest_categorical('activation_ext', ['relu', 'gelu', 'swish']);
		const cache = trial.suggest_bool('cache_ext');
		return Math.pow((lr - 0.008) / 0.0025, 2) + Math.pow((dropout - 0.12) / 0.05, 2) + Math.pow((hidden - 160) / 24, 2) + (activation === 'gelu' ? 0 : activation === 'swish' ? 2 : 6) + (cache ? 0 : 1.5);
	};
	const seeds = [1, 3, 7, 11, 21, 42, 66, 99, 256, 512, 1024, 2026];
	const losses = [];
	for (const seed of seeds) {
		const run = await runOptimizer(objective, { n: 180, seed });
		losses.push(run.bestLoss);
		console.log(`seed=${seed}, best_loss=${formatNumber(run.bestLoss)}, params=${JSON.stringify(run.bestParams)}`);
	}
	const averageLoss = mean(losses);
	const worstLoss = Math.max(...losses);
	console.log(`扩展平均损失: ${formatNumber(averageLoss)}`);
	console.log(`扩展最差损失: ${formatNumber(worstLoss)}`);
	return worstLoss < 0.1 && averageLoss < 0.02;
};

const baseScenarios = [
	{
		name: '单峰精度压测',
		description: '模拟单一核心业务指标调优，检查能否快速逼近解析最优解。',
		n: 120,
		seed: 7,
		baselineLoss: 49,
		targetLoss: 0,
		passRate: 0.999999,
		objective: async (trial) => {
			const bidMultiplier = trial.suggest_float('bid_multiplier', -4, 10);
			return Math.pow(bidMultiplier - 3, 2);
		},
		inspect: (bestParams, bestLoss) => ({
			ok: Math.abs(bestParams.bid_multiplier - 3) < 1e-4 && bestLoss < 1e-8,
			message: `bid_multiplier=${formatNumber(bestParams.bid_multiplier)}`,
		}),
	},
	{
		name: '业务混合空间压测',
		description: '模拟学习率、层数、批大小、优化器和布尔开关共同作用的真实调参面。',
		n: 260,
		seed: 42,
		baselineLoss: 143.1649,
		targetLoss: 0,
		passRate: 0.985,
		objective: async (trial) => {
			const lr = trial.suggest_float('lr', 0.0001, 0.05);
			const dropout = trial.suggest_float('dropout', 0, 0.6);
			const depth = trial.suggest_int('depth', 2, 8);
			const batch = trial.suggest_categorical('batch', [16, 32, 64, 128, 256]);
			const optimizer = trial.suggest_categorical('optimizer', ['sgd', 'adam', 'adamw']);
			const warmup = trial.suggest_bool('warmup');
			const labelSmoothing = trial.suggest_float('label_smoothing', 0, 0.2);
			const optimizerPenalty = { sgd: 10, adam: 0, adamw: 4 }[optimizer];
			const batchPenalty = { 16: 18, 32: 7, 64: 0, 128: 6, 256: 17 }[batch];
			const warmupPenalty = warmup ? 0 : 9;
			const interactionPenalty = warmup && optimizer === 'adam' ? 0 : 5;
			return Math.pow((lr - 0.012) / 0.0025, 2) + Math.pow((dropout - 0.18) / 0.06, 2) + Math.pow(depth - 5, 2) * 2 + Math.pow((labelSmoothing - 0.08) / 0.03, 2) + optimizerPenalty + batchPenalty + warmupPenalty + interactionPenalty;
		},
		inspect: (bestParams) => {
			const checks = [bestParams.optimizer === 'adam', bestParams.batch === 64, bestParams.warmup === true, Math.abs(bestParams.depth - 5) <= 1];
			return {
				ok: checks.filter(Boolean).length >= 4,
				message: `optimizer=${bestParams.optimizer}, batch=${bestParams.batch}, depth=${bestParams.depth}, warmup=${bestParams.warmup}`,
			};
		},
	},
	{
		name: '高维资源编排压测',
		description: '模拟 24 维连续资源位点联动，检查高维收敛与步长控制。',
		n: 360,
		seed: 11,
		baselineLoss: 305.75,
		targetLoss: 0,
		passRate: 0.93,
		objective: async (trial) => {
			let loss = 0;
			for (let i = 0; i < 24; i++) {
				const target = (i % 4) * 1.75 - 2.5;
				const value = trial.suggest_float(`channel_${i}`, -8, 8);
				loss += Math.pow((value - target) / 1.6, 2);
			}
			return loss;
		},
		inspect: (bestParams, bestLoss) => ({
			ok: bestLoss < 15,
			message: ['channel_0', 'channel_5', 'channel_13', 'channel_22'].map((key) => `${key}=${formatNumber(bestParams[key])}`).join(', '),
		}),
	},
	{
		name: '边界命中压测',
		description: '模拟最优解贴边的资源约束问题，检查边界裁剪后能否继续收敛。',
		n: 160,
		seed: 21,
		baselineLoss: 370,
		targetLoss: 0,
		passRate: 0.998,
		objective: async (trial) => {
			const quotaRatio = trial.suggest_float('quota_ratio', 0, 1);
			const threads = trial.suggest_int('threads', 1, 64);
			const reserve = trial.suggest_float('reserve', 0, 50);
			return Math.pow(quotaRatio - 1, 2) * 120 + Math.pow(threads - 64, 2) * 0.08 + Math.pow(reserve, 2) * 0.12;
		},
		inspect: (bestParams, bestLoss) => ({
			ok: bestParams.quota_ratio > 0.98 && bestParams.threads >= 63 && bestParams.reserve < 1 && bestLoss < 0.5,
			message: `quota_ratio=${formatNumber(bestParams.quota_ratio)}, threads=${bestParams.threads}, reserve=${formatNumber(bestParams.reserve)}`,
		}),
	},
	{
		name: '异步噪声鲁棒性压测',
		description: '模拟有轻微线上抖动的异步评估任务，检验噪声环境下的可用性。',
		n: 240,
		seed: 99,
		baselineLoss: 65.5,
		targetLoss: 0,
		passRate: 0.9,
		objective: async (trial) => {
			const serveLr = trial.suggest_float('serve_lr', 0.0005, 0.03);
			const threshold = trial.suggest_float('threshold', 0.1, 0.95);
			const shard = trial.suggest_int('shard', 1, 24);
			const cache = trial.suggest_bool('cache');
			await new Promise((resolve) => setTimeout(resolve, 2));
			const baseLoss = Math.pow((serveLr - 0.009) / 0.003, 2) + Math.pow((threshold - 0.63) / 0.08, 2) + Math.pow((shard - 12) / 3.5, 2) + (cache ? 0 : 4);
			const noise = createDeterministicNoise([serveLr, threshold, shard, cache ? 1 : 0], 0.3);
			return Math.max(0, baseLoss + noise);
		},
		inspect: (bestParams, bestLoss) => ({
			ok: bestLoss < 5 && bestParams.cache === true,
			message: `serve_lr=${formatNumber(bestParams.serve_lr)}, threshold=${formatNumber(bestParams.threshold)}, shard=${bestParams.shard}, cache=${bestParams.cache}`,
		}),
	},
	{
		name: '强耦合峡谷压测',
		description: '模拟参数间强耦合的狭窄可行域，检查仅靠独立 sigma 时的退化风险。',
		n: 320,
		seed: 123,
		baselineLoss: 280,
		targetLoss: 0,
		passRate: 0.85,
		objective: async (trial) => {
			const cpu = trial.suggest_float('cpu_ratio', 0, 1);
			const mem = trial.suggest_float('memory_ratio', 0, 1);
			const io = trial.suggest_float('io_ratio', 0, 1);
			return Math.pow((cpu + mem - 1.18) / 0.04, 2) + Math.pow((mem + io - 1.06) / 0.04, 2) + Math.pow((cpu - io - 0.08) / 0.025, 2);
		},
		inspect: (bestParams, bestLoss) => ({
			ok: bestLoss < 6,
			message: `cpu=${formatNumber(bestParams.cpu_ratio)}, mem=${formatNumber(bestParams.memory_ratio)}, io=${formatNumber(bestParams.io_ratio)}`,
		}),
	},
	{
		name: '冷启动稀疏参数压测',
		description: '模拟首轮只暴露少量参数、后续逐步解锁更多配置位点的业务冷启动过程。',
		n: 260,
		seed: 512,
		baselineLoss: 180,
		targetLoss: 0,
		passRate: 0.9,
		objective: async (trial) => {
			const stage = trial.suggest_int('stage', 1, 3);
			const baseWidth = trial.suggest_int('base_width', 64, 320);
			let loss = Math.pow((stage - 2) / 0.8, 2) + Math.pow((baseWidth - 192) / 28, 2);
			if (stage >= 2) {
				const expertWidth = trial.suggest_int('expert_width', 32, 192);
				const residual = trial.suggest_bool('residual_gate');
				loss += Math.pow((expertWidth - 96) / 22, 2) + (residual ? 0 : 5);
			}
			if (stage >= 3) {
				const auxLoss = trial.suggest_float('aux_loss_weight', 0, 0.4);
				const router = trial.suggest_categorical('router', ['top1', 'top2', 'sinkhorn']);
				loss += Math.pow((auxLoss - 0.18) / 0.05, 2) + (router === 'top2' ? 0 : router === 'sinkhorn' ? 3 : 7);
			}
			return loss;
		},
		inspect: (bestParams, bestLoss) => ({
			ok: bestLoss < 8 && bestParams.stage >= 2 && bestParams.base_width >= 160,
			message: `stage=${bestParams.stage}, base_width=${bestParams.base_width}, expert_width=${bestParams.expert_width}, router=${bestParams.router}`,
		}),
	},
	{
		name: '离散主导套餐压测',
		description: '模拟核心收益主要由离散套餐决定，连续参数只做微调的定价搜索问题。',
		n: 220,
		seed: 77,
		baselineLoss: 120,
		targetLoss: 0,
		passRate: 0.92,
		objective: async (trial) => {
			const packageTier = trial.suggest_categorical('package_tier', ['free', 'standard', 'growth', 'enterprise']);
			const region = trial.suggest_categorical('region', ['cn', 'sg', 'eu']);
			const retry = trial.suggest_int('retry_limit', 0, 8);
			const timeout = trial.suggest_float('timeout_factor', 0.5, 2.5);
			const packagePenalty = { free: 30, standard: 12, growth: 0, enterprise: 8 }[packageTier];
			const regionPenalty = { cn: 5, sg: 0, eu: 9 }[region];
			return packagePenalty + regionPenalty + Math.pow((retry - 3) / 1.2, 2) + Math.pow((timeout - 1.4) / 0.18, 2);
		},
		inspect: (bestParams, bestLoss) => ({
			ok: bestLoss < 3 && bestParams.package_tier === 'growth' && bestParams.region === 'sg',
			message: `package_tier=${bestParams.package_tier}, region=${bestParams.region}, retry_limit=${bestParams.retry_limit}, timeout_factor=${formatNumber(bestParams.timeout_factor)}`,
		}),
	},
	{
		name: '非零最优基线压测',
		description: '模拟存在不可消除业务噪声底座的场景，检查改善率计算在非零 target 下是否稳定。',
		n: 180,
		seed: 888,
		baselineLoss: 46,
		targetLoss: 7,
		passRate: 0.95,
		objective: async (trial) => {
			const price = trial.suggest_float('price_factor', 0.5, 1.5);
			const exposure = trial.suggest_float('exposure_bias', -2, 2);
			return 7 + Math.pow((price - 1.08) / 0.06, 2) + Math.pow((exposure + 0.35) / 0.14, 2);
		},
		inspect: (bestParams, bestLoss) => ({
			ok: bestLoss < 7.5,
			message: `price_factor=${formatNumber(bestParams.price_factor)}, exposure_bias=${formatNumber(bestParams.exposure_bias)}`,
		}),
	},
	{
		name: '窄范围整数抖动压测',
		description: '模拟最优点落在很窄的整数窗口内，检查四舍五入后的整数搜索精度。',
		n: 200,
		seed: 63,
		baselineLoss: 80,
		targetLoss: 0,
		passRate: 0.97,
		objective: async (trial) => {
			const workers = trial.suggest_int('workers', 28, 36);
			const microBatch = trial.suggest_int('micro_batch', 1, 6);
			const queueDepth = trial.suggest_int('queue_depth', 4, 14);
			return Math.pow(workers - 33, 2) * 2.5 + Math.pow(microBatch - 4, 2) * 3 + Math.pow(queueDepth - 9, 2);
		},
		inspect: (bestParams, bestLoss) => ({
			ok: bestLoss === 0 && bestParams.workers === 33 && bestParams.micro_batch === 4 && bestParams.queue_depth === 9,
			message: `workers=${bestParams.workers}, micro_batch=${bestParams.micro_batch}, queue_depth=${bestParams.queue_depth}`,
		}),
	},
	{
		name: '布尔门控组合压测',
		description: '模拟多个布尔开关与连续参数共同决定收益的线上策略开关选择问题。',
		n: 260,
		seed: 4096,
		baselineLoss: 95,
		targetLoss: 0,
		passRate: 0.9,
		objective: async (trial) => {
			const cache = trial.suggest_bool('cache_enabled');
			const prefetch = trial.suggest_bool('prefetch_enabled');
			const quant = trial.suggest_bool('quant_enabled');
			const ratio = trial.suggest_float('rerank_ratio', 0, 1);
			const penalty = (cache ? 0 : 8) + (prefetch ? 0 : 5) + (quant ? 6 : 0);
			const comboPenalty = cache && prefetch && !quant ? 0 : 4;
			return penalty + comboPenalty + Math.pow((ratio - 0.72) / 0.08, 2);
		},
		inspect: (bestParams, bestLoss) => ({
			ok: bestLoss < 2 && bestParams.cache_enabled === true && bestParams.prefetch_enabled === true && bestParams.quant_enabled === false,
			message: `cache=${bestParams.cache_enabled}, prefetch=${bestParams.prefetch_enabled}, quant=${bestParams.quant_enabled}, rerank_ratio=${formatNumber(bestParams.rerank_ratio)}`,
		}),
	},
	{
		name: '多峰陷阱压测',
		description: '模拟局部最优较多的损失地形，检查动量与步长扩缩能否跳出浅层陷阱。',
		n: 340,
		seed: 2718,
		baselineLoss: 160,
		targetLoss: 0,
		passRate: 0.82,
		objective: async (trial) => {
			const x = trial.suggest_float('x', -6, 6);
			const y = trial.suggest_float('y', -6, 6);
			return Math.pow(x - 1.8, 2) + Math.pow(y + 2.2, 2) + 1.6 * (1 - Math.cos(2.5 * x)) + 1.3 * (1 - Math.cos(2 * y));
		},
		inspect: (bestParams, bestLoss) => ({
			ok: bestLoss < 2.5,
			message: `x=${formatNumber(bestParams.x)}, y=${formatNumber(bestParams.y)}`,
		}),
	},
	{
		name: '超高维轻耦合压测',
		description: '模拟 60 维轻耦合参数面，检查参数数量暴增后的退化速度。',
		n: 520,
		seed: 135,
		baselineLoss: 800,
		targetLoss: 0,
		passRate: 0.88,
		objective: async (trial) => {
			let loss = 0;
			let sumA = 0;
			let sumB = 0;
			for (let i = 0; i < 60; i++) {
				const target = ((i % 6) - 2.5) * 0.9;
				const value = trial.suggest_float(`w_${i}`, -5, 5);
				loss += Math.pow((value - target) / 1.4, 2);
				if (i < 30) sumA += value;
				else sumB += value;
			}
			loss += Math.pow((sumA - 2.7) / 5.5, 2) + Math.pow((sumB + 1.8) / 5.5, 2);
			return loss;
		},
		inspect: (bestParams, bestLoss) => ({
			ok: bestLoss < 40,
			message: `w_0=${formatNumber(bestParams.w_0)}, w_17=${formatNumber(bestParams.w_17)}, w_34=${formatNumber(bestParams.w_34)}, w_59=${formatNumber(bestParams.w_59)}`,
		}),
	},
	{
		name: '冷热切换恢复压测',
		description: '模拟线上流量结构突变后重新收敛，检查在非平稳目标下的恢复能力。',
		n: 280,
		seed: 2025,
		baselineLoss: 140,
		targetLoss: 0,
		passRate: 0.78,
		objective: async (trial) => {
			const lr = trial.suggest_float('adaptive_lr', 0.001, 0.05);
			const mix = trial.suggest_float('traffic_mix', 0, 1);
			const shard = trial.suggest_int('serving_shard', 4, 20);
			const coldStartBoost = trial.suggest_bool('cold_start_boost');
			const targetLr = mix < 0.45 ? 0.011 : 0.026;
			const targetShard = mix < 0.45 ? 8 : 15;
			const boostPenalty = mix < 0.45 ? (coldStartBoost ? 3 : 0) : (coldStartBoost ? 0 : 4);
			return Math.pow((lr - targetLr) / 0.004, 2) + Math.pow((shard - targetShard) / 1.8, 2) + Math.pow((mix - 0.63) / 0.16, 2) + boostPenalty;
		},
		inspect: (bestParams, bestLoss) => ({
			ok: bestLoss < 8,
			message: `adaptive_lr=${formatNumber(bestParams.adaptive_lr)}, traffic_mix=${formatNumber(bestParams.traffic_mix)}, serving_shard=${bestParams.serving_shard}, cold_start_boost=${bestParams.cold_start_boost}`,
		}),
	},
	{
		name: '分类翻转敏感压测',
		description: '模拟分类选择一旦选错就造成大额损失，检查离散翻转能力。',
		n: 260,
		seed: 5501,
		baselineLoss: 150,
		targetLoss: 0,
		passRate: 0.92,
		objective: async (trial) => {
			const policy = trial.suggest_categorical('policy', ['baseline', 'explore', 'balanced', 'strict']);
			const temp = trial.suggest_float('temperature', 0.1, 1.3);
			const quota = trial.suggest_int('quota_bucket', 2, 10);
			const policyPenalty = { baseline: 16, explore: 9, balanced: 0, strict: 25 }[policy];
			return policyPenalty + Math.pow((temp - 0.62) / 0.09, 2) + Math.pow((quota - 6) / 1.2, 2);
		},
		inspect: (bestParams, bestLoss) => ({
			ok: bestLoss < 2 && bestParams.policy === 'balanced',
			message: `policy=${bestParams.policy}, temperature=${formatNumber(bestParams.temperature)}, quota_bucket=${bestParams.quota_bucket}`,
		}),
	},
	{
		name: '边界细粒度逼近压测',
		description: '模拟最优点极度贴近边界且容忍区间很窄，检查连续参数末端逼近能力。',
		n: 260,
		seed: 6012,
		baselineLoss: 120,
		targetLoss: 0,
		passRate: 0.94,
		objective: async (trial) => {
			const edgeX = trial.suggest_float('edge_x', 0, 1);
			const edgeY = trial.suggest_float('edge_y', -1, 1);
			return Math.pow((edgeX - 0.9972) / 0.0025, 2) + Math.pow((edgeY + 0.0008) / 0.015, 2);
		},
		inspect: (bestParams, bestLoss) => ({
			ok: bestLoss < 0.2 && bestParams.edge_x > 0.99,
			message: `edge_x=${formatNumber(bestParams.edge_x)}, edge_y=${formatNumber(bestParams.edge_y)}`,
		}),
	},
	{
		name: '整数台阶收敛压测',
		description: '模拟整数配置位点对收益呈台阶分布，检查离散整数的稳定命中能力。',
		n: 220,
		seed: 7301,
		baselineLoss: 110,
		targetLoss: 0,
		passRate: 0.95,
		objective: async (trial) => {
			const replicas = trial.suggest_int('replicas', 1, 9);
			const queue = trial.suggest_int('queue_slots', 2, 12);
			return Math.pow(replicas - 7, 2) * 2 + Math.pow(queue - 5, 2);
		},
		inspect: (bestParams, bestLoss) => ({
			ok: bestLoss === 0 && bestParams.replicas === 7 && bestParams.queue_slots === 5,
			message: `replicas=${bestParams.replicas}, queue_slots=${bestParams.queue_slots}`,
		}),
	},
	{
		name: '布尔偏置恢复压测',
		description: '模拟单一布尔开关带来显著收益，检查成功经验能否在后续试验中被放大。',
		n: 220,
		seed: 8119,
		baselineLoss: 70,
		targetLoss: 0,
		passRate: 0.96,
		objective: async (trial) => {
			const enableCache = trial.suggest_bool('enable_cache');
			const gate = trial.suggest_float('gate_ratio', 0, 1);
			return (enableCache ? 0 : 14) + Math.pow((gate - 0.41) / 0.07, 2);
		},
		inspect: (bestParams, bestLoss) => ({
			ok: bestLoss < 1 && bestParams.enable_cache === true,
			message: `enable_cache=${bestParams.enable_cache}, gate_ratio=${formatNumber(bestParams.gate_ratio)}`,
		}),
	},
	{
		name: '离散跨度稳态压测',
		description: '模拟分类与整数跨度较大时的稳定搜索，检查范围感知下离散步长是否仍足够活跃。',
		n: 260,
		seed: 9007,
		baselineLoss: 160,
		targetLoss: 0,
		passRate: 0.95,
		objective: async (trial) => {
			const mode = trial.suggest_categorical('mode', ['a', 'b', 'c', 'd', 'e', 'f']);
			const bucket = trial.suggest_int('bucket', 1, 20);
			const score = trial.suggest_float('score', 0, 1);
			const modePenalty = { a: 18, b: 14, c: 9, d: 0, e: 7, f: 21 }[mode];
			return modePenalty + Math.pow(bucket - 14, 2) * 0.7 + Math.pow((score - 0.58) / 0.08, 2);
		},
		inspect: (bestParams, bestLoss) => ({
			ok: bestLoss < 1 && bestParams.mode === 'd' && bestParams.bucket === 14,
			message: `mode=${bestParams.mode}, bucket=${bestParams.bucket}, score=${formatNumber(bestParams.score)}`,
		}),
	},
	{
		name: '批内二次采样压测',
		description: '模拟首个候选较差、二次微扰候选更优的业务面，检查批内补采样是否能提升命中率。',
		n: 240,
		seed: 9103,
		baselineLoss: 100,
		targetLoss: 0,
		passRate: 0.96,
		objective: async (trial) => {
			const x = trial.suggest_float('retry_x', -4, 4);
			const y = trial.suggest_float('retry_y', -4, 4);
			return Math.pow((x - 1.35) / 0.18, 2) + Math.pow((y + 1.1) / 0.18, 2);
		},
		inspect: (bestParams, bestLoss) => ({
			ok: bestLoss < 0.2,
			message: `retry_x=${formatNumber(bestParams.retry_x)}, retry_y=${formatNumber(bestParams.retry_y)}`,
		}),
	},
	{
		name: '近优噪声复验压测',
		description: '模拟候选接近最优时伴随轻微噪声，检查接近 best_loss 时的复验平均是否更稳。',
		n: 220,
		seed: 10021,
		baselineLoss: 80,
		targetLoss: 0,
		passRate: 0.95,
		objective: async (trial) => {
			const alpha = trial.suggest_float('noise_alpha', -2, 2);
			const beta = trial.suggest_float('noise_beta', -2, 2);
			const base = Math.pow((alpha - 0.45) / 0.12, 2) + Math.pow((beta + 0.35) / 0.12, 2);
			const noise = createDeterministicNoise([alpha, beta, base], 0.08);
			return Math.max(0, base + noise);
		},
		inspect: (bestParams, bestLoss) => ({
			ok: bestLoss < 0.15,
			message: `noise_alpha=${formatNumber(bestParams.noise_alpha)}, noise_beta=${formatNumber(bestParams.noise_beta)}`,
		}),
	},
	{
		name: '停滞恢复压测',
		description: '模拟长时间没有显著改进后需要重新放大探索半径，检查停滞恢复是否有效。',
		n: 320,
		seed: 11017,
		baselineLoss: 140,
		targetLoss: 0,
		passRate: 0.9,
		objective: async (trial) => {
			const x = trial.suggest_float('stall_x', -6, 6);
			const y = trial.suggest_float('stall_y', -6, 6);
			const ridge = Math.pow((x - 2.4) / 0.35, 2) + Math.pow((y + 2.2) / 0.35, 2);
			const trap = Math.exp(-Math.pow((x + 0.4) / 0.7, 2) - Math.pow((y - 0.3) / 0.7, 2)) * 4;
			return ridge + trap;
		},
		inspect: (bestParams, bestLoss) => ({
			ok: bestLoss < 1.5,
			message: `stall_x=${formatNumber(bestParams.stall_x)}, stall_y=${formatNumber(bestParams.stall_y)}`,
		}),
	},
	{
		name: '超长链路混合压测',
		description: '模拟连续、整数、布尔、分类在一条长链路中共同作用，验证复杂组合下的终态精度。',
		n: 360,
		seed: 12001,
		baselineLoss: 220,
		targetLoss: 0,
		passRate: 0.95,
		objective: async (trial) => {
			const lr = trial.suggest_float('long_lr', 0.0001, 0.03);
			const dropout = trial.suggest_float('long_dropout', 0, 0.4);
			const layers = trial.suggest_int('long_layers', 2, 10);
			const heads = trial.suggest_int('long_heads', 2, 16);
			const activation = trial.suggest_categorical('long_activation', ['relu', 'gelu', 'swish']);
			const cache = trial.suggest_bool('long_cache');
			const warmup = trial.suggest_bool('long_warmup');
			return Math.pow((lr - 0.0065) / 0.0018, 2) + Math.pow((dropout - 0.08) / 0.03, 2) + Math.pow((layers - 6) / 1.2, 2) + Math.pow((heads - 8) / 1.5, 2) + (activation === 'gelu' ? 0 : activation === 'swish' ? 1.5 : 5) + (cache ? 0 : 2.5) + (warmup ? 0 : 1.5);
		},
		inspect: (bestParams, bestLoss) => ({
			ok: bestLoss < 0.5 && bestParams.long_activation === 'gelu' && bestParams.long_cache === true,
			message: `long_lr=${formatNumber(bestParams.long_lr)}, long_layers=${bestParams.long_layers}, long_heads=${bestParams.long_heads}, long_activation=${bestParams.long_activation}`,
		}),
	},
	{
		name: '双布尔缓存策略压测',
		description: '模拟本地缓存与远端缓存两级联动，检查双布尔组合在连续参数陪衬下的命中能力。',
		n: 240,
		seed: 13003,
		baselineLoss: 120,
		targetLoss: 0,
		passRate: 0.96,
		objective: async (trial) => {
			const localCache = trial.suggest_bool('local_cache');
			const remoteCache = trial.suggest_bool('remote_cache');
			const refresh = trial.suggest_float('refresh_ratio', 0, 1);
			const penalty = (localCache ? 0 : 5) + (remoteCache ? 0 : 8) + (localCache && remoteCache ? 0 : 4);
			return penalty + Math.pow((refresh - 0.34) / 0.06, 2);
		},
		inspect: (bestParams, bestLoss) => ({
			ok: bestLoss < 0.5 && bestParams.local_cache === true && bestParams.remote_cache === true,
			message: `local_cache=${bestParams.local_cache}, remote_cache=${bestParams.remote_cache}, refresh_ratio=${formatNumber(bestParams.refresh_ratio)}`,
		}),
	},
	{
		name: '召回阈值整定压测',
		description: '模拟召回阈值与候选池大小协同调优，检查连续+整数的精细整定能力。',
		n: 260,
		seed: 14009,
		baselineLoss: 130,
		targetLoss: 0,
		passRate: 0.96,
		objective: async (trial) => {
			const threshold = trial.suggest_float('recall_threshold', 0.1, 0.95);
			const topK = trial.suggest_int('recall_topk', 10, 200);
			const rerank = trial.suggest_bool('recall_rerank');
			return Math.pow((threshold - 0.58) / 0.04, 2) + Math.pow((topK - 96) / 8, 2) + (rerank ? 0 : 2.5);
		},
		inspect: (bestParams, bestLoss) => ({
			ok: bestLoss < 0.2 && bestParams.recall_rerank === true,
			message: `recall_threshold=${formatNumber(bestParams.recall_threshold)}, recall_topk=${bestParams.recall_topk}, recall_rerank=${bestParams.recall_rerank}`,
		}),
	},
	{
		name: '分片副本协同压测',
		description: '模拟分片数与副本数共同影响吞吐与稳定性，检查整数协同搜索能力。',
		n: 260,
		seed: 15011,
		baselineLoss: 150,
		targetLoss: 0,
		passRate: 0.96,
		objective: async (trial) => {
			const shards = trial.suggest_int('cluster_shards', 2, 20);
			const replicas = trial.suggest_int('cluster_replicas', 1, 6);
			const cache = trial.suggest_bool('cluster_cache');
			return Math.pow((shards - 12) / 1.4, 2) + Math.pow((replicas - 3) / 0.8, 2) + (cache ? 0 : 1.2);
		},
		inspect: (bestParams, bestLoss) => ({
			ok: bestLoss < 0.2 && bestParams.cluster_shards === 12 && bestParams.cluster_replicas === 3,
			message: `cluster_shards=${bestParams.cluster_shards}, cluster_replicas=${bestParams.cluster_replicas}, cluster_cache=${bestParams.cluster_cache}`,
		}),
	},
	{
		name: '分类路由精确压测',
		description: '模拟路由策略选择一旦偏差就明显降质，检查分类变量在长链路中的精确命中。',
		n: 260,
		seed: 16013,
		baselineLoss: 140,
		targetLoss: 0,
		passRate: 0.96,
		objective: async (trial) => {
			const route = trial.suggest_categorical('route_policy', ['random', 'latency', 'quality', 'hybrid']);
			const score = trial.suggest_float('route_score', 0, 1);
			const routePenalty = { random: 9, latency: 5, quality: 0, hybrid: 3 }[route];
			return routePenalty + Math.pow((score - 0.67) / 0.05, 2);
		},
		inspect: (bestParams, bestLoss) => ({
			ok: bestLoss < 0.2 && bestParams.route_policy === 'quality',
			message: `route_policy=${bestParams.route_policy}, route_score=${formatNumber(bestParams.route_score)}`,
		}),
	},
	{
		name: '预算分配单峰压测',
		description: '模拟预算系数与探索比例的单峰调优，检查连续参数在业务预算面上的精细逼近。',
		n: 240,
		seed: 17021,
		baselineLoss: 120,
		targetLoss: 0,
		passRate: 0.97,
		objective: async (trial) => {
			const budget = trial.suggest_float('budget_ratio', 0.2, 1.4);
			const explore = trial.suggest_float('explore_ratio', 0, 0.5);
			return Math.pow((budget - 0.92) / 0.04, 2) + Math.pow((explore - 0.11) / 0.03, 2);
		},
		inspect: (bestParams, bestLoss) => ({
			ok: bestLoss < 0.05,
			message: `budget_ratio=${formatNumber(bestParams.budget_ratio)}, explore_ratio=${formatNumber(bestParams.explore_ratio)}`,
		}),
	},
	{
		name: '预热边界压测',
		description: '模拟预热步数贴近边界的场景，检查边界附近整数与连续混合参数的稳定命中。',
		n: 240,
		seed: 18031,
		baselineLoss: 130,
		targetLoss: 0,
		passRate: 0.97,
		objective: async (trial) => {
			const warmupSteps = trial.suggest_int('warmup_steps', 0, 40);
			const decay = trial.suggest_float('decay_ratio', 0.6, 1);
			return Math.pow((warmupSteps - 40) / 1.2, 2) + Math.pow((decay - 0.97) / 0.015, 2);
		},
		inspect: (bestParams, bestLoss) => ({
			ok: bestLoss < 0.05 && bestParams.warmup_steps === 40,
			message: `warmup_steps=${bestParams.warmup_steps}, decay_ratio=${formatNumber(bestParams.decay_ratio)}`,
		}),
	},
	{
		name: '资源档位选择压测',
		description: '模拟资源档位选择与连续伸缩参数联动，检查分类+连续组合的稳定命中。',
		n: 240,
		seed: 19027,
		baselineLoss: 120,
		targetLoss: 0,
		passRate: 0.97,
		objective: async (trial) => {
			const tier = trial.suggest_categorical('resource_tier', ['small', 'medium', 'large', 'xlarge']);
			const scale = trial.suggest_float('resource_scale', 0.5, 1.5);
			const tierPenalty = { small: 8, medium: 3, large: 0, xlarge: 5 }[tier];
			return tierPenalty + Math.pow((scale - 1.08) / 0.05, 2);
		},
		inspect: (bestParams, bestLoss) => ({
			ok: bestLoss < 0.1 && bestParams.resource_tier === 'large',
			message: `resource_tier=${bestParams.resource_tier}, resource_scale=${formatNumber(bestParams.resource_scale)}`,
		}),
	},
	{
		name: '轻噪声离散收敛压测',
		description: '模拟轻微噪声下的离散档位选择，检查分类变量在抖动目标上的收敛能力。',
		n: 240,
		seed: 20033,
		baselineLoss: 120,
		targetLoss: 0,
		passRate: 0.97,
		objective: async (trial) => {
			const mode = trial.suggest_categorical('noisy_mode', ['cold', 'warm', 'hot']);
			const ratio = trial.suggest_float('noisy_ratio', 0, 1);
			const modePenalty = { cold: 6, warm: 0, hot: 4 }[mode];
			const noise = createDeterministicNoise([ratio, modePenalty], 0.03);
			return Math.max(0, modePenalty + Math.pow((ratio - 0.46) / 0.05, 2) + noise);
		},
		inspect: (bestParams, bestLoss) => ({
			ok: bestLoss < 0.1 && bestParams.noisy_mode === 'warm',
			message: `noisy_mode=${bestParams.noisy_mode}, noisy_ratio=${formatNumber(bestParams.noisy_ratio)}`,
		}),
	},
	{
		name: '三元布尔策略压测',
		description: '模拟三个布尔开关共同决定策略收益，检查多开关组合在连续参数陪衬下的稳定命中。',
		n: 260,
		seed: 21037,
		baselineLoss: 150,
		targetLoss: 0,
		passRate: 0.97,
		objective: async (trial) => {
			const cache = trial.suggest_bool('tri_cache');
			const prefetch = trial.suggest_bool('tri_prefetch');
			const compact = trial.suggest_bool('tri_compact');
			const ratio = trial.suggest_float('tri_ratio', 0, 1);
			const penalty = (cache ? 0 : 4) + (prefetch ? 0 : 3) + (compact ? 0 : 2) + (cache && prefetch && compact ? 0 : 5);
			return penalty + Math.pow((ratio - 0.52) / 0.04, 2);
		},
		inspect: (bestParams, bestLoss) => ({
			ok: bestLoss < 0.1 && bestParams.tri_cache === true && bestParams.tri_prefetch === true && bestParams.tri_compact === true,
			message: `tri_cache=${bestParams.tri_cache}, tri_prefetch=${bestParams.tri_prefetch}, tri_compact=${bestParams.tri_compact}, tri_ratio=${formatNumber(bestParams.tri_ratio)}`,
		}),
	},
	{
		name: '小轮数小参数压测',
		description: '覆盖最低轮数与最小参数量组合，验证 10 轮下 2 参数连续面是否仍能快速收敛。',
		n: 10,
		seed: 22043,
		baselineLoss: 60,
		targetLoss: 0,
		passRate: 0.7,
		objective: async (trial) => {
			const alpha = trial.suggest_float('tiny_alpha', -4, 4);
			const beta = trial.suggest_float('tiny_beta', -4, 4);
			return Math.pow(alpha / 0.45, 2) + Math.pow(beta / 0.45, 2);
		},
		inspect: (bestParams, bestLoss) => ({
			ok: bestLoss < 1e-10,
			message: `tiny_alpha=${formatNumber(bestParams.tiny_alpha)}, tiny_beta=${formatNumber(bestParams.tiny_beta)}`,
		}),
	},
	{
		name: '小轮数大参数压测',
		description: '覆盖 10 轮配合 100 参数的大维度连续面，验证高维早期预算下的有效搜索半径。',
		n: 10,
		seed: 23047,
		baselineLoss: 900,
		targetLoss: 0,
		passRate: 0.3,
		objective: async (trial) => {
			let loss = 0;
			for (let i = 0; i < 100; i++) {
				const value = trial.suggest_float(`short_high_dim_${i}`, -3, 3);
				loss += Math.pow(value / 1.4, 2);
			}
			return loss;
		},
		inspect: (bestParams, bestLoss) => ({
			ok: bestLoss < 1e-10,
			message: `short_high_dim_0=${formatNumber(bestParams.short_high_dim_0)}, short_high_dim_49=${formatNumber(bestParams.short_high_dim_49)}, short_high_dim_99=${formatNumber(bestParams.short_high_dim_99)}`,
		}),
	},
	{
		name: '大轮数小参数压测',
		description: '覆盖 1000 轮配合 2 参数，验证长时间迭代下连续精度不会被过度探索破坏。',
		n: 1000,
		seed: 24053,
		baselineLoss: 120,
		targetLoss: 0,
		passRate: 0.9999,
		objective: async (trial) => {
			const x = trial.suggest_float('long_small_x', -8, 8);
			const y = trial.suggest_float('long_small_y', -8, 8);
			return Math.pow((x - 2.345) / 0.08, 2) + Math.pow((y + 1.765) / 0.08, 2);
		},
		inspect: (bestParams, bestLoss) => ({
			ok: bestLoss < 1e-6,
			message: `long_small_x=${formatNumber(bestParams.long_small_x)}, long_small_y=${formatNumber(bestParams.long_small_y)}`,
		}),
	},
	{
		name: '大轮数大参数混合压测',
		description: '覆盖 1000 轮与 100 参数混合类型，验证超长预算下连续、整数、布尔、分类的联合收敛。',
		n: 1000,
		seed: 25061,
		baselineLoss: 1600,
		targetLoss: 0,
		passRate: 0.97,
		objective: async (trial) => {
			let loss = 0;
			const profileTargets = ['compact', 'balanced', 'throughput', 'latency'];
			for (let i = 0; i < 40; i++) {
				const value = trial.suggest_float(`mix_float_${i}`, -4, 4);
				loss += Math.pow(value / 0.8, 2);
			}
			for (let i = 0; i < 25; i++) {
				const value = trial.suggest_int(`mix_int_${i}`, 1, 24);
				loss += Math.pow((value - 13) / 1.1, 2);
			}
			for (let i = 0; i < 20; i++) {
				const enabled = trial.suggest_bool(`mix_bool_${i}`);
				loss += enabled ? 2.2 : 0;
			}
			for (let i = 0; i < 15; i++) {
				const profile = trial.suggest_categorical(`mix_cat_${i}`, profileTargets);
				loss += profile === 'throughput' ? 0 : 2.8;
			}
			return loss;
		},
		inspect: (bestParams, bestLoss) => ({
			ok: bestLoss < 1e-10 && bestParams.mix_int_7 === 13 && bestParams.mix_bool_5 === false && bestParams.mix_cat_9 === 'throughput',
			message: `mix_float_0=${formatNumber(bestParams.mix_float_0)}, mix_int_7=${bestParams.mix_int_7}, mix_bool_5=${bestParams.mix_bool_5}, mix_cat_9=${bestParams.mix_cat_9}`,
		}),
	},
	{
		name: '中轮数中参数全类型压测',
		description: '覆盖中等轮数与 48 参数的全类型组合，验证 float/int/bool/categorical 在均衡预算下的协同效果。',
		n: 180,
		seed: 26071,
		baselineLoss: 520,
		targetLoss: 0,
		passRate: 0.92,
		objective: async (trial) => {
			let loss = 0;
			for (let i = 0; i < 18; i++) {
				const value = trial.suggest_float(`balanced_float_${i}`, -3, 3);
				loss += Math.pow(value / 0.7, 2);
			}
			for (let i = 0; i < 12; i++) {
				const value = trial.suggest_int(`balanced_int_${i}`, 1, 12);
				loss += Math.pow((value - 7) / 0.8, 2);
			}
			for (let i = 0; i < 9; i++) {
				const enabled = trial.suggest_bool(`balanced_bool_${i}`);
				loss += enabled ? 2 : 0;
			}
			for (let i = 0; i < 9; i++) {
				const mode = trial.suggest_categorical(`balanced_cat_${i}`, ['a', 'b', 'c']);
				loss += mode === 'b' ? 0 : 2.4;
			}
			return loss;
		},
		inspect: (bestParams, bestLoss) => ({
			ok: bestLoss < 1e-10 && bestParams.balanced_int_3 === 7 && bestParams.balanced_bool_4 === false && bestParams.balanced_cat_8 === 'b',
			message: `balanced_float_0=${formatNumber(bestParams.balanced_float_0)}, balanced_int_3=${bestParams.balanced_int_3}, balanced_bool_4=${bestParams.balanced_bool_4}, balanced_cat_8=${bestParams.balanced_cat_8}`,
		}),
	},
	{
		name: '小轮数大参数离散混合压测',
		description: '覆盖 20 轮下的大参数离散主导组合，验证少预算时整数、布尔、分类的有效翻转能力。',
		n: 20,
		seed: 27073,
		baselineLoss: 780,
		targetLoss: 0,
		passRate: 0.4,
		objective: async (trial) => {
			let loss = 0;
			for (let i = 0; i < 30; i++) {
				const value = trial.suggest_int(`short_mix_int_${i}`, 1, 20);
				loss += Math.pow(value - 11, 2) * 0.5;
			}
			for (let i = 0; i < 30; i++) {
				const enabled = trial.suggest_bool(`short_mix_bool_${i}`);
				loss += enabled ? 2 : 0;
			}
			for (let i = 0; i < 20; i++) {
				const choice = trial.suggest_categorical(`short_mix_cat_${i}`, ['cold', 'warm', 'hot', 'burst']);
				loss += choice === 'hot' ? 0 : 2.5;
			}
			for (let i = 0; i < 20; i++) {
				const value = trial.suggest_float(`short_mix_float_${i}`, -2, 2);
				loss += Math.pow(value / 1.1, 2);
			}
			return loss;
		},
		inspect: (bestParams, bestLoss) => ({
			ok: bestLoss < 1e-10 && bestParams.short_mix_int_0 === 11 && bestParams.short_mix_bool_7 === false && bestParams.short_mix_cat_3 === 'hot',
			message: `short_mix_int_0=${bestParams.short_mix_int_0}, short_mix_bool_7=${bestParams.short_mix_bool_7}, short_mix_cat_3=${bestParams.short_mix_cat_3}, short_mix_float_19=${formatNumber(bestParams.short_mix_float_19)}`,
		}),
	},
	{
		name: '中轮数超大参数连续压测',
		description: '覆盖 300 轮下 100 连续参数，验证中长预算对超大连续空间的稳定压缩能力。',
		n: 300,
		seed: 28081,
		baselineLoss: 1350,
		targetLoss: 0,
		passRate: 0.9,
		objective: async (trial) => {
			let loss = 0;
			for (let i = 0; i < 100; i++) {
				const value = trial.suggest_float(`mid_huge_${i}`, -3, 3);
				loss += Math.pow(value / 0.8, 2);
			}
			return loss;
		},
		inspect: (bestParams, bestLoss) => ({
			ok: bestLoss < 1e-10,
			message: `mid_huge_0=${formatNumber(bestParams.mid_huge_0)}, mid_huge_25=${formatNumber(bestParams.mid_huge_25)}, mid_huge_75=${formatNumber(bestParams.mid_huge_75)}, mid_huge_99=${formatNumber(bestParams.mid_huge_99)}`,
		}),
	},
	{
		name: '大轮数布尔分类网络压测',
		description: '覆盖长轮数下多布尔与多分类网络，验证长期迭代不会丢失离散结构命中能力。',
		n: 720,
		seed: 29083,
		baselineLoss: 980,
		targetLoss: 0,
		passRate: 0.97,
		objective: async (trial) => {
			let loss = 0;
			for (let i = 0; i < 30; i++) {
				const flag = trial.suggest_bool(`long_disc_bool_${i}`);
				loss += flag ? 2.2 : 0;
			}
			for (let i = 0; i < 20; i++) {
				const option = trial.suggest_categorical(`long_disc_cat_${i}`, ['cpu', 'mem', 'net', 'io', 'mix']);
				loss += option === 'net' ? 0 : 2.8;
			}
			for (let i = 0; i < 10; i++) {
				const ratio = trial.suggest_float(`long_disc_float_${i}`, 0, 1);
				loss += Math.pow((ratio - 0.5) / 0.03, 2);
			}
			return loss;
		},
		inspect: (bestParams, bestLoss) => ({
			ok: bestLoss < 1e-10 && bestParams.long_disc_bool_0 === false && bestParams.long_disc_cat_4 === 'net',
			message: `long_disc_bool_0=${bestParams.long_disc_bool_0}, long_disc_cat_4=${bestParams.long_disc_cat_4}, long_disc_float_9=${formatNumber(bestParams.long_disc_float_9)}`,
		}),
	},
	{
		name: '轮数梯度恢复压测',
		description: '覆盖 60 轮与 64 参数的分层目标，验证中小预算下参数子集更新策略对混合空间的恢复效果。',
		n: 60,
		seed: 30089,
		baselineLoss: 620,
		targetLoss: 0,
		passRate: 0.82,
		objective: async (trial) => {
			let loss = 0;
			for (let i = 0; i < 24; i++) {
				const value = trial.suggest_float(`ladder_float_${i}`, -2.5, 2.5);
				loss += Math.pow(value / 0.6, 2);
			}
			for (let i = 0; i < 16; i++) {
				const value = trial.suggest_int(`ladder_int_${i}`, 1, 16);
				loss += Math.pow((value - 9) / 1.1, 2);
			}
			for (let i = 0; i < 12; i++) {
				const enabled = trial.suggest_bool(`ladder_bool_${i}`);
				loss += enabled ? 2.1 : 0;
			}
			for (let i = 0; i < 12; i++) {
				const choice = trial.suggest_categorical(`ladder_cat_${i}`, ['p0', 'p1', 'p2', 'p3']);
				loss += choice === 'p2' ? 0 : 2.5;
			}
			return loss;
		},
		inspect: (bestParams, bestLoss) => ({
			ok: bestLoss < 1e-10 && bestParams.ladder_int_7 === 9 && bestParams.ladder_bool_5 === false && bestParams.ladder_cat_11 === 'p2',
			message: `ladder_float_0=${formatNumber(bestParams.ladder_float_0)}, ladder_int_7=${bestParams.ladder_int_7}, ladder_bool_5=${bestParams.ladder_bool_5}, ladder_cat_11=${bestParams.ladder_cat_11}`,
		}),
	},
];

const extraScenarios = [
	{
		name: '有效步长缩放压测',
		description: '验证有效变化维度才放大或缩小步长后，高维静默参数不会拖累主变量收敛。',
		n: 180,
		seed: 31001,
		baselineLoss: 180,
		targetLoss: 0,
		passRate: 0.98,
		objective: async (trial) => {
			const focus_x = trial.suggest_float('focus_x', -6, 6);
			const focus_y = trial.suggest_float('focus_y', -6, 6);
			for (let i = 0; i < 12; i++) {
				trial.suggest_float(`silent_dim_${i}`, -20, 20);
			}
			return Math.pow(focus_x - 1.75, 2) + Math.pow(focus_y + 2.25, 2);
		},
		inspect: (bestParams, bestLoss) => ({
			ok: Math.abs(bestParams.focus_x - 1.75) < 1e-4 && Math.abs(bestParams.focus_y + 2.25) < 1e-4 && bestLoss < 1e-8,
			message: `focus_x=${formatNumber(bestParams.focus_x)}, focus_y=${formatNumber(bestParams.focus_y)}`,
		}),
	},
	{
		name: '邻域重试细化压测',
		description: '验证失败后围绕首候选做局部细化时，窄谷连续面能够更稳定命中。',
		n: 220,
		seed: 32003,
		baselineLoss: 150,
		targetLoss: 0,
		passRate: 0.99,
		objective: async (trial) => {
			const retry_focus_x = trial.suggest_float('retry_focus_x', -3, 3);
			const retry_focus_y = trial.suggest_float('retry_focus_y', -3, 3);
			return Math.pow((retry_focus_x - 0.375) / 0.25, 2) + Math.pow((retry_focus_y + 0.625) / 0.18, 2);
		},
		inspect: (bestParams, bestLoss) => ({
			ok: Math.abs(bestParams.retry_focus_x - 0.375) < 5e-4 && Math.abs(bestParams.retry_focus_y + 0.625) < 5e-4 && bestLoss < 1e-4,
			message: `retry_focus_x=${formatNumber(bestParams.retry_focus_x)}, retry_focus_y=${formatNumber(bestParams.retry_focus_y)}`,
		}),
	},
	{
		name: '边界缩步回弹压测',
		description: '验证超界后缩步重采样时，贴边连续与整数参数仍能继续向边界最优逼近。',
		n: 220,
		seed: 33007,
		baselineLoss: 220,
		targetLoss: 0,
		passRate: 0.995,
		objective: async (trial) => {
			const clamp_ratio = trial.suggest_float('clamp_ratio', 0, 1);
			const clamp_threads = trial.suggest_int('clamp_threads', 8, 32);
			return Math.pow(clamp_ratio - 1, 2) * 140 + Math.pow(clamp_threads - 32, 2) * 0.35;
		},
		inspect: (bestParams, bestLoss) => ({
			ok: bestParams.clamp_ratio > 0.999 && bestParams.clamp_threads === 32 && bestLoss < 0.01,
			message: `clamp_ratio=${formatNumber(bestParams.clamp_ratio)}, clamp_threads=${bestParams.clamp_threads}`,
		}),
	},
	{
		name: '改进幅度布尔偏置压测',
		description: '验证布尔参数在带来大收益时会获得更强偏置记忆，并带动连续参数快速贴近最优。',
		n: 220,
		seed: 34011,
		baselineLoss: 140,
		targetLoss: 0,
		passRate: 0.99,
		objective: async (trial) => {
			const strong_gate = trial.suggest_bool('strong_gate');
			const strong_ratio = trial.suggest_float('strong_ratio', 0, 1);
			return (strong_gate ? 0 : 45) + Math.pow((strong_ratio - 0.68) / 0.04, 2);
		},
		inspect: (bestParams, bestLoss) => ({
			ok: bestParams.strong_gate === true && Math.abs(bestParams.strong_ratio - 0.68) < 5e-4 && bestLoss < 1e-5,
			message: `strong_gate=${bestParams.strong_gate}, strong_ratio=${formatNumber(bestParams.strong_ratio)}`,
		}),
	},
	{
		name: '新最优复验裁决压测',
		description: '验证仅在逼近新最优时做复验平均，轻噪声场景下仍能稳定接受真正更优候选。',
		n: 240,
		seed: 35013,
		baselineLoss: 130,
		targetLoss: 0,
		passRate: 0.98,
		objective: async (trial) => {
			const confirm_alpha = trial.suggest_float('confirm_alpha', -2, 2);
			const confirm_beta = trial.suggest_float('confirm_beta', -2, 2);
			const noise = createDeterministicNoise([confirm_alpha, confirm_beta], 0.00004);
			return Math.pow(confirm_alpha - 0.42, 2) + Math.pow(confirm_beta + 0.31, 2) + noise;
		},
		inspect: (bestParams, bestLoss) => ({
			ok: Math.abs(bestParams.confirm_alpha - 0.42) < 0.004 && Math.abs(bestParams.confirm_beta + 0.311) < 0.004 && bestLoss < 0.002,
			message: `confirm_alpha=${formatNumber(bestParams.confirm_alpha)}, confirm_beta=${formatNumber(bestParams.confirm_beta)}`,
		}),
	},
	{
		name: '分类整数协同压测',
		description: '扩展离散测试覆盖，验证分类与整数同时命中时连续参数也能完成精细对齐。',
		n: 240,
		seed: 36017,
		baselineLoss: 180,
		targetLoss: 0,
		passRate: 0.99,
		objective: async (trial) => {
			const route_pack = trial.suggest_categorical('route_pack', ['basic', 'plus', 'max']);
			const route_slots = trial.suggest_int('route_slots', 2, 10);
			const route_ratio = trial.suggest_float('route_ratio', 0.2, 1.2);
			return (route_pack === 'plus' ? 0 : 18) + Math.pow(route_slots - 7, 2) + Math.pow((route_ratio - 0.74) / 0.03, 2);
		},
		inspect: (bestParams, bestLoss) => ({
			ok: bestParams.route_pack === 'plus' && bestParams.route_slots === 7 && Math.abs(bestParams.route_ratio - 0.74) < 5e-4 && bestLoss < 1e-5,
			message: `route_pack=${bestParams.route_pack}, route_slots=${bestParams.route_slots}, route_ratio=${formatNumber(bestParams.route_ratio)}`,
		}),
	},
	{
		name: '静默维度隔离压测',
		description: '继续扩展高维覆盖，验证大量无贡献维度存在时核心目标仍可快速归零。',
		n: 200,
		seed: 37019,
		baselineLoss: 240,
		targetLoss: 0,
		passRate: 0.995,
		objective: async (trial) => {
			let loss = 0;
			for (let i = 0; i < 4; i++) {
				const value = trial.suggest_float(`active_iso_${i}`, -5, 5);
				loss += Math.pow(value, 2);
			}
			for (let i = 0; i < 20; i++) {
				trial.suggest_float(`inactive_iso_${i}`, -9, 9);
			}
			return loss;
		},
		inspect: (bestParams, bestLoss) => ({
			ok: bestLoss < 1e-8 && Math.abs(bestParams.active_iso_0) < 1e-4 && Math.abs(bestParams.active_iso_3) < 1e-4,
			message: `active_iso_0=${formatNumber(bestParams.active_iso_0)}, active_iso_3=${formatNumber(bestParams.active_iso_3)}`,
		}),
	},
	{
		name: '低预算边界离散压测',
		description: '覆盖低预算下的边界整数、布尔与分类联合命中，验证短轮数 mixed space 稳定性。',
		n: 36,
		seed: 38023,
		baselineLoss: 160,
		targetLoss: 0,
		passRate: 0.92,
		objective: async (trial) => {
			const budget_int = trial.suggest_int('budget_int', 1, 12);
			const budget_gate = trial.suggest_bool('budget_gate');
			const budget_mode = trial.suggest_categorical('budget_mode', ['cold', 'warm', 'hot']);
			return Math.pow(budget_int - 12, 2) * 0.4 + (budget_gate ? 0 : 12) + (budget_mode === 'hot' ? 0 : 9);
		},
		inspect: (bestParams, bestLoss) => ({
			ok: bestParams.budget_int === 12 && bestParams.budget_gate === true && bestParams.budget_mode === 'hot' && bestLoss < 1e-8,
			message: `budget_int=${bestParams.budget_int}, budget_gate=${bestParams.budget_gate}, budget_mode=${bestParams.budget_mode}`,
		}),
	},
	{
		name: '双布尔连续共振压测',
		description: '扩展双布尔与连续变量联合命中验证，检查偏置记忆不会拖慢连续精修。',
		n: 240,
		seed: 39029,
		baselineLoss: 170,
		targetLoss: 0,
		passRate: 0.99,
		objective: async (trial) => {
			const resonance_cache = trial.suggest_bool('resonance_cache');
			const resonance_prefetch = trial.suggest_bool('resonance_prefetch');
			const resonance_ratio = trial.suggest_float('resonance_ratio', 0, 1);
			return (resonance_cache ? 0 : 16) + (resonance_prefetch ? 0 : 14) + Math.pow((resonance_ratio - 0.57) / 0.025, 2);
		},
		inspect: (bestParams, bestLoss) => ({
			ok: bestParams.resonance_cache === true && bestParams.resonance_prefetch === true && Math.abs(bestParams.resonance_ratio - 0.57) < 5e-4 && bestLoss < 1e-5,
			message: `resonance_cache=${bestParams.resonance_cache}, resonance_prefetch=${bestParams.resonance_prefetch}, resonance_ratio=${formatNumber(bestParams.resonance_ratio)}`,
		}),
	},
	{
		name: '温和多峰补偿压测',
		description: '补充温和多峰面测试，验证局部细化与有效步长控制在多峰场景下仍能稳定命中主谷。',
		n: 260,
		seed: 40031,
		baselineLoss: 150,
		targetLoss: 0,
		passRate: 0.97,
		objective: async (trial) => {
			const mellow_x = trial.suggest_float('mellow_x', -4, 4);
			const mellow_y = trial.suggest_float('mellow_y', -4, 4);
			const basin = Math.pow(mellow_x - 1.1, 2) + Math.pow(mellow_y + 1.4, 2);
			const ripple = 0.08 * (Math.sin(3 * mellow_x) + Math.cos(2 * mellow_y) + 2);
			return basin + ripple;
		},
		inspect: (bestParams, bestLoss) => ({
			ok: Math.abs(bestParams.mellow_x - 1.207) < 0.03 && Math.abs(bestParams.mellow_y + 1.423) < 0.03 && bestLoss < 0.07,
			message: `mellow_x=${formatNumber(bestParams.mellow_x)}, mellow_y=${formatNumber(bestParams.mellow_y)}`,
		}),
	},
];

const extraScenarioCount = Math.max(0, Math.min(extraScenarios.length, Number(process.env.EXTRA_SCENARIOS ?? extraScenarios.length)));
const scenarios = [...baseScenarios, ...extraScenarios.slice(0, extraScenarioCount)];

console.log('超参数优化器高压力业务测试开始');
const results = [];
for (const scenario of scenarios) {
	results.push(await evaluateScenario(scenario));
}
for (const result of results) printScenario(result);

const passCount = results.filter((result) => result.pass).length;
const avgImprove = mean(results.map((result) => result.improvementRate));
const avgLoss = mean(results.map((result) => result.bestLoss));

console.log('\n=== 场景汇总 ===');
console.log(`通过场景: ${passCount}/${results.length}`);
console.log(`平均改善率: ${(avgImprove * 100).toFixed(2)}%`);
console.log(`平均 best_loss: ${formatNumber(avgLoss)}`);

const resumeOk = await evaluateResumeConsistency();
const dynamicSchemaResumeOk = await evaluateDynamicSchemaResumeConsistency();
const nonPrimitiveResumeOk = await evaluateNonPrimitiveCategoricalResumeConsistency();
const nonPrimitiveResetResumeOk = await evaluateNonPrimitiveCategoricalResetAfterResume();
const nonPrimitiveMutationIsolationOk = await evaluateNonPrimitiveCategoricalMutationIsolation();
const cacheRestoreIsolationOk = await evaluateCacheRestoreIsolation();
const metaRegistrationOk = await evaluateMetaRegistrationStability();
const deterministicOk = await evaluateDeterminism();
const seedOk = await evaluateSeedSensitivity();
const extendedSeedOk = await evaluateExtendedSeedStability();

console.log('\n=== 综合结论 ===');
console.log(`断点续跑一致性: ${resumeOk ? 'PASS' : 'WARN'}`);
console.log(`动态 Schema 续跑一致性: ${dynamicSchemaResumeOk ? 'PASS' : 'WARN'}`);
console.log(`非 Primitive 分类续跑一致性: ${nonPrimitiveResumeOk ? 'PASS' : 'WARN'}`);
console.log(`非 Primitive 分类 reset 续跑一致性: ${nonPrimitiveResetResumeOk ? 'PASS' : 'WARN'}`);
console.log(`非 Primitive 分类引用隔离一致性: ${nonPrimitiveMutationIsolationOk ? 'PASS' : 'WARN'}`);
console.log(`cache 恢复引用隔离一致性: ${cacheRestoreIsolationOk ? 'PASS' : 'WARN'}`);
console.log(`元信息注册稳定性: ${metaRegistrationOk ? 'PASS' : 'WARN'}`);
console.log(`同种子复现性: ${deterministicOk ? 'PASS' : 'WARN'}`);
console.log(`多种子稳定性: ${seedOk ? 'PASS' : 'WARN'}`);
console.log(`扩展多种子基准: ${extendedSeedOk ? 'PASS' : 'WARN'}`);
console.log(`总体评级: ${passCount === results.length && resumeOk && dynamicSchemaResumeOk && nonPrimitiveResumeOk && nonPrimitiveResetResumeOk && nonPrimitiveMutationIsolationOk && cacheRestoreIsolationOk && metaRegistrationOk && deterministicOk && seedOk && extendedSeedOk ? '可用于业务压测基线' : '存在可观测短板，建议继续优化算法'}`);
console.log(`SUMMARY_JSON:${JSON.stringify({
	passCount,
	totalCount: results.length,
	avgImprove,
	avgLoss,
	resumeOk,
	dynamicSchemaResumeOk,
	nonPrimitiveResumeOk,
	nonPrimitiveResetResumeOk,
	nonPrimitiveMutationIsolationOk,
	cacheRestoreIsolationOk,
	metaRegistrationOk,
	deterministicOk,
	seedOk,
	extendedSeedOk,
})}`);

if (process.env.RUN_ISOLATED_BENCHMARKS === '1') {
	await evaluateIsolatedMicroBenchmarks();
}
