import { Worker } from 'worker_threads';


// 超时功能 - 10秒后自动结束
setTimeout(() => {
	console.log('运行超时退出');
	process.exit();
}, 1000 * 10);

const worker = new Worker('./test.js');

worker.on('exit', (code) => {
	console.log('执行完毕');
	process.exit();
});


