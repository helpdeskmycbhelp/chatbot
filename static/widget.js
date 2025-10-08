(() => {
  // Where is your API? default = same origin; override via data-api on the script tag
  const API = (document.currentScript?.dataset?.api) || window.location.origin;
  const CHAT_URL = `${API.replace(/\/$/,'')}/static/chat.html?api=${encodeURIComponent(API)}`;

  // Shadow root to avoid CSS conflicts
  const r = document.createElement('div');
  document.body.appendChild(r);
  const shadow = r.attachShadow({ mode: 'open' });

  const css = document.createElement('style');
  css.textContent = `
    .wrap{position:fixed;right:20px;bottom:20px;z-index:999999}
    .btn{width:56px;height:56px;border-radius:999px;background:#2563eb;color:#fff;border:0;cursor:pointer;
         display:grid;place-items:center;font:600 16px system-ui;box-shadow:0 10px 30px rgba(0,0,0,.18)}
    .panel{position:fixed;right:20px;bottom:86px;width:360px;height:520px;background:#fff;border-radius:16px;
           box-shadow:0 18px 50px rgba(0,0,0,.22);overflow:hidden;display:none}
    .panel.open{display:block}
    .head{height:46px;background:#0f172a;color:#fff;display:flex;align-items:center;justify-content:space-between;padding:0 12px;font:600 14px system-ui}
    .close{background:transparent;border:0;color:#fff;font-size:18px;cursor:pointer}
    iframe{width:100%;height:calc(100% - 46px);border:0}
    @media (max-width:480px){.panel{right:0;bottom:0;width:100vw;height:100svh;border-radius:0}}
  `;

  const wrap = document.createElement('div'); wrap.className = 'wrap';
  const btn  = document.createElement('button'); btn.className = 'btn'; btn.textContent = '💬';
  const panel= document.createElement('div'); panel.className = 'panel';
  const head = document.createElement('div'); head.className = 'head'; head.innerHTML = '<span>Policy Assistant</span>';
  const x    = document.createElement('button'); x.className = 'close'; x.textContent = '✕';
  const frame= document.createElement('iframe'); frame.src = CHAT_URL;

  head.appendChild(x);
  panel.appendChild(head);
  panel.appendChild(frame);
  wrap.appendChild(panel);
  wrap.appendChild(btn);
  shadow.appendChild(css);
  shadow.appendChild(wrap);

  btn.onclick = () => panel.classList.toggle('open');
  x.onclick   = () => panel.classList.remove('open');
})();
