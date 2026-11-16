/* Human-readable editor; imported turns stay unchanged until explicitly edited. */
(function () {
  const byId = id => document.getElementById(id);
  const jsonInput = byId('input'), conversation = byId('conversation');
  let previous = null, previousText = '', syncing = false;
  const roles = { client: 'Клиент', operator: 'Оператор' };
  function show(message) { byId('message').textContent = message; }
  function render(call) {
    if (!call || !Array.isArray(call.turns)) return;
    previous = call;
    previousText = call.turns.map(t => `${roles[t.speaker] || t.speaker}: ${t.text}`).join('\n');
    conversation.value = previousText;
    byId('call-id').value = call.call_id || '';
    byId('call-date').value = call.started_at || '';
  }
  function fromJson() {
    if (syncing) return;
    try { render(JSON.parse(jsonInput.value)); } catch (_) { /* core editor reports invalid JSON */ }
  }
  function parseTurns(text) {
    const turns = [];
    for (const line of text.split(/\r?\n/)) {
      const match = line.match(/^\s*(Клиент|Оператор|client|operator): ?(.*)$/i);
      if (match) {
        turns.push({ idx: turns.length, speaker: /^(Клиент|client)$/i.test(match[1]) ? 'client' : 'operator', text: match[2] });
      } else if (turns.length) { turns[turns.length - 1].text += '\n' + line; }
      else if (line.trim()) { throw Error('Укажите роль перед первой репликой: Клиент: или Оператор:'); }
    }
    if (!turns.length) throw Error('Добавьте хотя бы одну реплику.');
    return turns;
  }
  function toJson() {
    try {
      const turns = previous && conversation.value === previousText ? previous.turns : parseTurns(conversation.value);
      const call = { ...(previous || {}), call_id: byId('call-id').value || 'manual-' + Date.now(),
        started_at: byId('call-date').value, turns };
      syncing = true;
      jsonInput.value = JSON.stringify(call, null, 2);
      jsonInput.dispatchEvent(new Event('input', { bubbles: true }));
    } catch (error) {
      // Never submit old valid input after a human-editor error.
      syncing = true;
      jsonInput.value = '';
      jsonInput.dispatchEvent(new Event('input', { bubbles: true }));
      show(error.message);
    } finally { syncing = false; }
  }
  jsonInput.addEventListener('input', fromJson);
  byId('example').addEventListener('change', fromJson);
  conversation.addEventListener('input', toJson);
  byId('call-id').addEventListener('input', toJson);
  byId('call-date').addEventListener('input', toJson);
  byId('upload-call').addEventListener('change', async event => {
    const file = event.target.files[0]; if (!file) return;
    try {
      if (file.size > 256 * 1024) throw Error('Файл слишком большой. Загрузите один звонок размером до 256 КБ.');
      const raw = await file.text(); let value;
      try { value = JSON.parse(raw); } catch (_) {
        const lines = raw.split(/\r?\n/).filter(s => s.trim());
        if (lines.length !== 1) throw Error('Загрузите один JSON-объект звонка, а не весь датасет.');
        value = JSON.parse(lines[0]);
      }
      if (!value || Array.isArray(value) || !Array.isArray(value.turns)) throw Error('В файле нужны call_id, started_at и turns.');
      const keys = ['call_id', 'started_at', 'duration_sec', 'series', 'turns', 'noise_level'];
      const call = Object.fromEntries(keys.filter(k => Object.hasOwn(value, k)).map(k => [k, value[k]]));
      syncing = true;
      jsonInput.value = JSON.stringify(call, null, 2);
      jsonInput.dispatchEvent(new Event('input', { bubbles: true }));
      render(call);
      show(Object.hasOwn(value, 'facts') ? 'Загружены только реплики и метаданные. Эталонные facts исключены из запроса.' : 'Звонок загружен. Нажмите «Извлечь факты».');
    } catch (error) { show('Ошибка загрузки: ' + error.message); }
    finally { syncing = false; event.target.value = ''; }
  });
  byId('download-output').addEventListener('click', () => {
    try {
      const value = JSON.parse(byId('json').textContent);
      if (!value.prediction) throw Error('Сначала получите результат извлечения.');
      const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], { type: 'application/json;charset=utf-8' }));
      const link = document.createElement('a'); link.href = url;
      link.download = 'extraction-result.json'; link.click(); URL.revokeObjectURL(url);
    } catch (error) { show(error.message); }
  });
  fromJson();
  window.compactFactEditor = { parseTurns };
})();
