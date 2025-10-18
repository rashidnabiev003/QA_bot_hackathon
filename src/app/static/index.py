HTML = """
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>QA Bot</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; padding: 0; background: #ffffff; color: #000000; }
    .wrap { display: grid; grid-template-columns: 1fr 1fr; height: 100vh; }
    .left, .right { padding: 12px; box-sizing: border-box; }
    .left { border-right: 1px solid #e0e0e0; display: flex; flex-direction: column; }
    .chat { flex: 1; overflow: auto; border: 1px solid #e0e0e0; padding: 10px; }
    .input { display: flex; gap: 8px; margin-top: 8px; }
    .input textarea { flex: 1; height: 60px; background: #f0f0f0; color: #000000; }
    .send { background: #10a37f; color: #fff; border: 0; padding: 8px 12px; }
    .ctx, .metrics { height: 50%; overflow: auto; border: 1px solid #e0e0e0; padding: 10px; background: #f9f9f9; }
  </style>
  <script>
    async function sendQuery() {
      const q = document.getElementById('q').value.trim();
      const useLLM = document.getElementById('llm').checked;
      if (!q) return;
      const chat = document.getElementById('chat');
      chat.innerHTML += `<div><b>Вы:</b> ${q}</div>`;
      document.getElementById('q').value = '';
      const res = await fetch('/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query: q, use_llm_rerank: useLLM }) });
      const data = await res.json();
      const ans = (data.answer && data.answer.answer) || 'Нет ответа';
      chat.innerHTML += `<div><b>AI:</b> ${ans}</div>`;
      const ctxDiv = document.getElementById('ctx');
      ctxDiv.innerHTML = '';
      (data.contexts || []).forEach((c, i) => {
        let scoreInfo = '';
        if (c.llm_score !== undefined) {
          scoreInfo = `<span style="color: #666;">LLM score: ${c.llm_score.toFixed(3)}</span><br/>`;
        } else if (c.score_rerank !== undefined) {
          scoreInfo = `<span style="color: #666;">rerank score: ${c.score_rerank.toFixed(3)}</span><br/>`;
        } else if (c.score !== undefined) {
          scoreInfo = `<span style="color: #666;">score: ${c.score.toFixed(3)}</span><br/>`;
        }
        ctxDiv.innerHTML += `<div style="margin-bottom: 10px; padding: 10px; border: 1px solid #ddd; border-radius: 5px;"><b>[${i+1}]</b> ${scoreInfo}<div style="margin-top: 5px;">${c.text}</div></div>`;
      });
      const m = data.metrics || {};
      const metrics = document.getElementById('metrics');
      metrics.innerHTML = '<h4>Метрики ответа:</h4>';
      Object.keys(m).forEach(k => {
        const val = typeof m[k] === 'number' ? m[k].toFixed(3) : m[k];
        metrics.innerHTML += `<div><b>${k}:</b> ${val}</div>`;
      });
    }
    async function runBenchmark() {
      const useLLM = document.getElementById('llm_bench').checked;
      const limitVal = document.getElementById('limit').value.trim();
      const payload = { use_llm_rerank: useLLM };
      if (limitVal) { payload.limit = parseInt(limitVal, 10); }
      const btn = document.getElementById('bench');
      const old = btn.textContent; btn.textContent = 'Запуск...'; btn.disabled = true;
      try {
        const res = await fetch('/benchmark', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
        const data = await res.json();
        alert(data.ok ? ('Сохранено: ' + data.path + ' (' + data.rows + ' строк)') : ('Ошибка: ' + data.error));
      } catch(e) {
        alert('Ошибка запуска бенчмарка: ' + e);
      } finally {
        btn.textContent = old; btn.disabled = false;
      }
    }
    window.addEventListener('DOMContentLoaded', () => {
      document.getElementById('send').addEventListener('click', sendQuery);
      const rebuildBtn = document.createElement('button');
      rebuildBtn.id = 'rebuild';
      rebuildBtn.className = 'send';
      rebuildBtn.style.background = '#dc3545';
      rebuildBtn.textContent = 'Пересобрать индекс';
      document.querySelector('.input').appendChild(rebuildBtn);
      rebuildBtn.addEventListener('click', async () => {
        rebuildBtn.disabled = true; const old = rebuildBtn.textContent; rebuildBtn.textContent = 'Пересборка...';
        try {
          const res = await fetch('/rebuild_index', { method: 'POST' });
          const data = await res.json();
          alert(data.ok ? 'Индекс пересобран' : ('Ошибка: ' + data.error));
        } finally {
          rebuildBtn.disabled = false; rebuildBtn.textContent = old;
        }
      });
      
      const benchWrap = document.createElement('div');
      benchWrap.style.marginTop = '8px';
      benchWrap.innerHTML = `
        <label><input type=\"checkbox\" id=\"llm_bench\"/> LLM-реранкинг (бенчмарк)</label>
        <input id=\"limit\" placeholder=\"limit\" style=\"width:80px; margin-left:6px;\"/>
        <button id=\"bench\" class=\"send\" style=\"background:#0d6efd\">Бенчмарк XLSX</button>
      `;
      document.querySelector('.input').appendChild(benchWrap);
      document.getElementById('bench').addEventListener('click', runBenchmark);
    });
  </script>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"left\">
      <div id=\"chat\" class=\"chat\"></div>
      <div class=\"input\">
        <textarea id=\"q\" placeholder=\"Введите вопрос...\"></textarea>
        <label><input type=\"checkbox\" id=\"llm\"/> LLM-реранкинг</label>
        <button id=\"send\" class=\"send\">Отправить</button>
      </div>
    </div>
    <div class=\"right\">
      <div id=\"ctx\" class=\"ctx\"></div>
      <div id=\"metrics\" class=\"metrics\"></div>
    </div>
  </div>
</body>
</html>
"""

def get_html():
    return HTML
