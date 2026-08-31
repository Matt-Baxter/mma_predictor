/**
 * Check that the browser implementation of the network agrees with PyTorch.
 *
 * The page re-implements the forward pass in plain JavaScript, so it could
 * silently drift from the Python model. This runs the real page script under a
 * stub DOM and compares its probabilities against predictions.json, which
 * scripts/verify_web.py writes straight from the trained PyTorch ensemble.
 */
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const root = path.resolve(__dirname, '..');
const html = fs.readFileSync(path.join(root, 'web/index.html'), 'utf8');
const scripts = [...html.matchAll(/<script(?![^>]*type="application\/json")[^>]*>([\s\S]*?)<\/script>/g)];
const source = scripts[scripts.length - 1][1];

const stub = () => new Proxy(function () {}, {
  get(target, prop) {
    if (prop === 'textContent' || prop === 'value' || prop === 'innerHTML') return '';
    if (prop === 'style') return {};
    if (prop === 'children') return [stub(), stub()];
    if (prop === 'checked') return false;
    if (prop === Symbol.toPrimitive) return () => '';
    return stub();
  },
  set() { return true; },
  apply(target, thisArg, args) {
    if (args[0] === '.fname') return [stub(), stub()];
    return stub();
  },
});

const context = {
  document: {
    getElementById: () => stub(),
    createElement: () => stub(),
  },
  fetch: async () => ({
    json: async () => JSON.parse(fs.readFileSync(path.join(root, 'web/model_data.json'), 'utf8')),
  }),
  setTimeout,
  console,
  Math,
  JSON,
  Object,
  Number,
  Float64Array,
  globalThis: null,
};
context.globalThis = context;
vm.createContext(context);
vm.runInContext(source, context);

setTimeout(() => {
  const expected = JSON.parse(fs.readFileSync(path.join(root, 'web/predictions.json'), 'utf8'));
  let worst = 0;
  let failures = 0;
  for (const row of expected) {
    const got = context.__predict(row.a, row.b, row.title).probability;
    const delta = Math.abs(got - row.p);
    worst = Math.max(worst, delta);
    const ok = delta < 1e-4;
    if (!ok) failures++;
    console.log(
      `  ${ok ? 'ok  ' : 'FAIL'} ${(row.a + ' vs ' + row.b).padEnd(46)} ` +
      `js=${got.toFixed(6)} py=${row.p.toFixed(6)} diff=${delta.toExponential(2)}`
    );
  }
  console.log(`\n  ${expected.length - failures}/${expected.length} match; worst difference ${worst.toExponential(2)}`);
  process.exit(failures ? 1 : 0);
}, 400);
