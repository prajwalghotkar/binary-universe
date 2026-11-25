(function () {
  var win = window.parent;
  var doc = win.document;
  var old = doc.getElementById('bin-rain');
  if (old) old.remove();
  if (win.__binRaf) win.cancelAnimationFrame(win.__binRaf);
  if (win.__binResize) win.removeEventListener('resize', win.__binResize);

  var canvas = doc.createElement('canvas');
  canvas.id = 'bin-rain';
  canvas.style.cssText = 'position:fixed;top:0;left:0;width:100vw;height:100vh;z-index:-1;pointer-events:none;opacity:0.5;';
  doc.body.appendChild(canvas);
  var ctx = canvas.getContext('2d');

  var size = 18;
  var angle = -0.3;
  var interval = 55;
  var w, h, span, cols, drops, stars, last = 0;

  function setup() {
    w = canvas.width = win.innerWidth;
    h = canvas.height = win.innerHeight;
    span = Math.hypot(w, h);
    cols = Math.ceil(span / size);
    drops = [];
    for (var i = 0; i < cols; i++) drops.push(Math.random() * -50);
    stars = [];
    for (var s = 0; s < 140; s++) {
      stars.push([Math.random() * w, Math.random() * h, Math.random() * 1.3 + 0.3, Math.random() * 6.28]);
    }
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, w, h);
  }

  function step(t) {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = 'rgba(0,0,0,0.09)';
    ctx.fillRect(0, 0, w, h);
    ctx.translate(w / 2, h / 2);
    ctx.rotate(angle);
    ctx.font = size + 'px monospace';
    ctx.fillStyle = '#8fb4ff';
    for (var i = 0; i < cols; i++) {
      var x = -span / 2 + i * size;
      var y = -span / 2 + drops[i] * size;
      ctx.fillText(Math.random() < 0.5 ? '0' : '1', x, y);
      if (y > span / 2 && Math.random() > 0.975) drops[i] = 0;
      drops[i] += 1;
    }
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    for (var k = 0; k < stars.length; k++) {
      var st = stars[k];
      ctx.globalAlpha = 0.35 + 0.35 * Math.sin(t / 900 + st[3]);
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(st[0], st[1], st[2], st[2]);
    }
    ctx.globalAlpha = 1;
  }

  function loop(t) {
    if (t - last > interval) {
      last = t;
      step(t);
    }
    win.__binRaf = win.requestAnimationFrame(loop);
  }

  win.__binResize = setup;
  win.addEventListener('resize', setup);
  setup();

  if (win.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    for (var n = 0; n < 160; n++) step(n * 55);
  } else {
    win.__binRaf = win.requestAnimationFrame(loop);
  }
})();
