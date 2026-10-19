"""Enter a shell with platform proxy authentication kept in process memory."""
import getpass
import os
from urllib.parse import quote

endpoint = input('平台代理地址（回车使用 10.16.1.51:3128）：').strip() or '10.16.1.51:3128'
username = input('平台代理用户名：').strip()
if not username:
    raise SystemExit('请输入平台提供的代理用户名')
password = getpass.getpass('平台代理密码（不显示）：')
if not password:
    raise SystemExit('请输入平台提供的代理密码')
proxy = f'http://{quote(username, safe="")}:{quote(password, safe="")}@{endpoint}'
env = os.environ.copy()
env.update(START_PROXY=proxy, HTTP_PROXY=proxy, HTTPS_PROXY=proxy,
           http_proxy=proxy, https_proxy=proxy, NO_PROXY='localhost,127.0.0.1',
           no_proxy='localhost,127.0.0.1', PS1='smolvla-start:\\w\\$ ')
print('代理已配置，进入操作终端；密码不会写入脚本或命令历史。输入 exit 离开。')
os.execvpe('bash', ['bash', '--noprofile', '--norc'], env)
