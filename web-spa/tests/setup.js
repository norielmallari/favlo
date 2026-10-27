// Expo installs a lazy structuredClone that loads @ungap/structured-clone.
// That package reads the global while the module is still evaluating.
Object.defineProperty(global, 'structuredClone', {
  configurable: true,
  writable: true,
  value(value) {
    return JSON.parse(JSON.stringify(value));
  },
});
