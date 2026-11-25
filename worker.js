// 在 Web Worker 里跑 Python（Pyodide），这样死循环不会卡住页面，主线程可以直接终止它。
var PYODIDE = 'https://cdn.jsdelivr.net/pyodide/v0.27.7/full/';
importScripts(PYODIDE + 'pyodide.js');

var ready = (async function () {
  var py = await loadPyodide({ indexURL: PYODIDE });
  await py.loadPackage('numpy');
  var res = await fetch('runner.py');
  if (!res.ok) throw new Error('runner.py 加载失败：' + res.status);
  py.runPython(await res.text());          // 定义 run(user_code, setup, tests)
  return py;
})();

ready.then(
  function () { postMessage({ type: 'ready' }); },
  function (err) { postMessage({ type: 'fatal', message: String(err && err.message || err) }); }
);

onmessage = async function (e) {
  var py;
  try { py = await ready; } catch (err) { return; }   // 已经报过 fatal
  var m = e.data, result;
  try {
    py.globals.set('_code', m.code);
    py.globals.set('_setup', m.setup);
    py.globals.set('_tests', JSON.stringify(m.tests));
    result = JSON.parse(py.runPython(
      'import json\njson.dumps(run(_code, _setup, json.loads(_tests)), ensure_ascii=False)'
    ));
  } catch (err) {
    result = { error: String(err && err.message || err), tests: [], stdout: '' };
  }
  postMessage({ type: 'result', id: m.id, result: result });
};
