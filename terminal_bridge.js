({url, command, start, end}) => {
    const address = new URL(url);
    if (address.origin !== location.origin) throw new Error('终端不在当前站点。');
    address.protocol = address.protocol === 'https:' ? 'wss:' : 'ws:';
    const s = {pending: '', buffer: '', started: false, done: false, code: null,
        error: null, sent: false, ws: new WebSocket(address.href)};
    window.__dcuTerminal = s;
    const marker = '\n' + end + ':';
    s.ws.onmessage = event => {
        let data;
        try { data = JSON.parse(event.data); } catch (_) { return; }
        if (data[0] === 'setup' && !s.sent) {
            s.sent = true;
            s.ws.send(JSON.stringify(['set_size', 40, 200, 0, 0]));
            s.ws.send(JSON.stringify(['stdin', command]));
        }
        if (data[0] === 'disconnect' && !s.done) {
            s.error = '服务器关闭了专用终端，命令完成状态未知。';
        }
        if (data[0] !== 'stdout' || s.done) return;
        // Strip only CR: retain ANSI sequences in logs rather than corrupting split sequences.
        s.buffer += data[1].replace(/\r/g, '');
        if (!s.started) {
            const index = s.buffer.indexOf('\n' + start + '\n');
            if (index < 0) {
                s.buffer = s.buffer.slice(-(start.length + 4));
                return;
            }
            s.buffer = s.buffer.slice(index + start.length + 2);
            s.started = true;
        }
        const index = s.buffer.indexOf(marker);
        if (index >= 0) {
            const tail = s.buffer.slice(index + marker.length);
            const match = tail.match(/^(\d+)\n/);
            if (match) {
                s.pending += s.buffer.slice(0, index);
                s.buffer = '';
                s.code = Number(match[1]);
                s.done = true;
            }
        } else {
            // Keep enough suffix to recognize a completion marker split across frames.
            const count = Math.max(0, s.buffer.length - marker.length - 16);
            s.pending += s.buffer.slice(0, count);
            s.buffer = s.buffer.slice(count);
        }
    };
    s.ws.onerror = () => { s.error = '终端 WebSocket 连接失败；检查登录和平台转发。'; };
    s.ws.onclose = () => {
        if (!s.done) {
            s.pending += s.buffer;
            s.buffer = '';
            s.error = s.error || '终端连接断开，命令完成状态未知；请勿直接重试有副作用的命令。';
        }
    };
}
