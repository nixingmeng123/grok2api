<div align="center">

# Grok2API 个人修改版

Grok 网页端能力的 OpenAI / Anthropic 兼容 API 网关

仓库地址：[nixingmeng123/grok2api](https://github.com/nixingmeng123/grok2api)

</div>

> [!NOTE]
> 本项目仅供学习与研究交流。请遵守 Grok / xAI 的使用条款及当地法律法规，不得用于非法用途。账号 token、API key、Cloudflare token 等敏感凭证不要提交到 GitHub，也不要发给别人。

## 项目来源

本仓库是 `nixingmeng123` 基于上游 [jiujiu532/grok2api](https://github.com/jiujiu532/grok2api) 修改维护的个人版本。上游 `jiujiu532/grok2api` 又基于 [chenyme/grok2api](https://github.com/chenyme/grok2api) 二次开发。

感谢 `jiujiu532` 和 `chenyme` 等原作者与维护者。本仓库保留上游项目的主要能力，并在此基础上加入个人适配。

## 本版本改动

- 新增 `grok-4.5-fast` 模型别名。
- `grok-4.5-fast` 映射到 Grok 网页端 `fast` 模式，即当前网页端显示的 Grok 4.5 Fast。
- 新增 `grok-4.5-console` / `grok-4.5-low` / `grok-4.5-medium` / `grok-4.5-high`。
- 4.5 thinking 系列走 `console.x.ai/v1/responses`，实际上游模型字段为 `grok-4.5`，并按模型名固定 `reasoning.effort`。
- `/v1/messages` 支持 Claude Code 原生工具调用，可使用 Read、Write、Edit、Glob、Grep、Bash、PowerShell 等客户端工具。
- Claude Code 的 WebSearch / WebFetch 交给 Grok 上游搜索处理，避免依赖客户端额外的安全分类器。
- 增加多轮工具结果转换、异常 XML 兼容和重复文件写入保护。
- `grok-imagine-video` 支持 Free/Basic 账号的 Console DPoP 视频额度，可进行文生视频和单图生视频。
- Console 视频使用异步轮询、SSE 心跳和防重复提交保护；内容审核拒绝会返回明确错误。
- 标准版 `docker-compose.yml` 已改为从本仓库源码构建，不再直接拉取原作者 `ghcr.io/jiujiu532/grok2api:latest` 镜像。
- 防封版 `docker-compose.warp.yml` 的 grok2api 主服务也已改为从本仓库源码构建。

> [!IMPORTANT]
> 如果部署时直接使用 `ghcr.io/jiujiu532/grok2api:latest`，部署到的是原作者镜像，不是本仓库修改版。要使用本版本，请从 `https://github.com/nixingmeng123/grok2api` 拉源码并 `docker build` / `docker compose up --build`。

## 快速部署

### 标准版

```bash
git clone https://github.com/nixingmeng123/grok2api.git
cd grok2api
cp .env.example .env
docker compose up -d --build
```

查看日志：

```bash
docker compose logs -f grok2api
```

### 防封版

适合服务器出口 IP 被 Cloudflare 拦截、需要 WARP / FlareSolverr 的情况。

```bash
git clone https://github.com/nixingmeng123/grok2api.git
cd grok2api
cp .env.example .env
docker compose -f docker-compose.warp.yml up -d --build
```

防封版里可能仍会看到这个辅助镜像：

```text
ghcr.io/jiujiu532/privoxy-warp:latest
```

它只是 WARP HTTP 代理辅助服务，不是 grok2api 主服务。grok2api 主服务会从本仓库源码构建。

### 单容器源码构建

```bash
git clone https://github.com/nixingmeng123/grok2api.git
cd grok2api
docker build -t grok2api:grok45 .

docker run -d \
  --name grok2api \
  --restart unless-stopped \
  -p 8000:8000 \
  -e TZ=Asia/Shanghai \
  -e LOG_LEVEL=INFO \
  -e ACCOUNT_STORAGE=local \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  grok2api:grok45
```

## 更新本版本

如果已经 clone 过本仓库：

```bash
cd grok2api
git pull
docker compose up -d --build
```

防封版更新：

```bash
cd grok2api
git pull
docker compose -f docker-compose.warp.yml up -d --build
```

## 首次启动

访问：

```text
http://服务器IP:8000/admin/login
```

默认密码通常为：

```text
grok2api
```

进入后台后建议配置：

| 配置项 | 说明 |
| :-- | :-- |
| `app.app_key` | Admin 后台密码 |
| `app.api_key` | API 调用密钥 |
| `app.app_url` | 公网访问地址，图片/视频链接需要，例如 `https://your-domain.com` |

配置保存后通常即时生效。

## 常用模型

| 模型名 | 路由 | 说明 |
| :-- | :-- | :-- |
| `grok-4.5-fast` | grok.com `fast` | 网页端 Grok 4.5 Fast 模式，速度快 |
| `grok-4.5-console` | console.x.ai | Console 版 Grok 4.5，默认低思考 |
| `grok-4.5-low` | console.x.ai | Grok 4.5，`reasoning.effort=low` |
| `grok-4.5-medium` | console.x.ai | Grok 4.5，`reasoning.effort=medium` |
| `grok-4.5-high` | console.x.ai | Grok 4.5，`reasoning.effort=high`，适合复杂代码/推理 |
| `grok-4.20-fast` | grok.com `fast` | 上游已有 fast 别名 |
| `grok-4.3-fast` | grok.com `fast` | 上游已有 fast 别名 |
| `grok-4.3-low` / `medium` / `high` | console.x.ai | Grok 4.3 thinking 系列 |
| `grok-4.20-auto` | grok.com `auto` | 需要对应账号等级 |
| `grok-4.20-expert` | grok.com `expert` | 需要对应账号等级 |
| `grok-4.20-multi-agent-console` | console.x.ai | Console 多智能体路由，可能更适合复杂推理，但更容易触发上游限制 |
| `grok-imagine-video` | console.x.ai | Free/Basic Console 视频，支持 6/10/12 秒文生视频和单图生视频 |

验证 `grok-4.5-fast` 是否走网页端 fast 路由：

```bash
docker exec -i grok2api python - <<'PY'
from app.control.model.registry import resolve
m = resolve("grok-4.5-fast")
print("model:", m.model_name)
print("mode_str:", m.mode_id.to_api_str())
print("is_console:", m.is_console_chat())
PY
```

期望输出包含：

```text
model: grok-4.5-fast
mode_str: fast
is_console: False
```

验证 `grok-4.5-high` 是否走 Console thinking 路由：

```bash
docker exec -i grok2api python - <<'PY'
from app.control.model.registry import resolve
from app.dataplane.reverse.protocol.xai_console_chat import build_console_payload
m = resolve("grok-4.5-high")
p = build_console_payload(messages=[{"role":"user","content":"hi"}], model="grok-4.5-high")
print("model:", m.model_name)
print("is_console:", m.is_console_chat())
print("upstream_model:", p["model"])
print("reasoning:", p.get("reasoning"))
PY
```

期望输出包含：

```text
model: grok-4.5-high
is_console: True
upstream_model: grok-4.5
reasoning: {'effort': 'high'}
```

## API 示例

OpenAI 兼容聊天接口：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"grok-4.5-fast","stream":true,"messages":[{"role":"user","content":"你好"}]}'
```

高思考强度示例：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"grok-4.5-high","stream":true,"messages":[{"role":"user","content":"写一个带测试的排序函数"}]}'
```

OpenAI Responses API 兼容接口：

```bash
curl http://localhost:8000/v1/responses \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"grok-4.5-high","stream":true,"input":"你好"}'
```

如果客户端支持设置思考强度，也可以传 `reasoning.effort`：

```bash
curl http://localhost:8000/v1/responses \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"grok-4.5-console","stream":true,"reasoning":{"effort":"high"},"input":"写一个带测试的排序函数"}'
```

Cherry Studio 可选 `OpenAI Compatible`，模式选择 `Responses API` 或 `OpenAI Responses`，Base URL 填到 `/v1` 结尾，例如 `https://your-domain.com/v1`，模型填 `grok-4.5-high`。

Console 视频示例：

```bash
curl http://localhost:8000/v1/videos \
  -H "Authorization: Bearer $API_KEY" \
  -F "model=grok-imagine-video" \
  -F "prompt=未来都市雨后夜景，镜头平稳向前移动" \
  -F "seconds=6" \
  -F "size=1280x720"
```

视频生成可能需要数分钟。流式 WebUI 会每 5 秒发送心跳，最长等待 15 分钟；拿到上游 `request_id` 后不会自动重新提交，以免重复消耗额度。内容审核由 xAI 上游执行，审核拒绝不是部署故障。

### Claude Code

Claude Code 使用 Anthropic Messages API，Base URL 填服务根地址，不要添加 `/v1`：

```text
http://服务器IP:8000
```

使用反向代理时：

```text
https://your-domain.com
```

推荐模型为 `grok-4.5-high`。本版本会把 Claude Code 提供的 Read、Write、Edit、Glob、Grep、Bash、PowerShell 等工具作为原生 function tools 转发，并将结果转换回 Anthropic `tool_use`。WebSearch / WebFetch 由 Grok 上游搜索处理。

文件读取、写入和命令执行发生在运行 Claude Code 的客户端机器上，不是在 grok2api 服务器容器内执行。客户端仍需授予相应目录权限。

API 基础地址格式：

```text
http://服务器IP:8000/v1
```

如果使用 Cloudflare Tunnel 或反向代理，则改成自己的域名：

```text
https://your-domain.com/v1
```

## 给 AI 部署时的提示词

可以把下面这段直接发给 AI 或朋友：

```text
请部署这个仓库源码版本：
https://github.com/nixingmeng123/grok2api

不要使用 ghcr.io/jiujiu532/grok2api:latest 镜像，因为那是原作者版本。
请从本仓库 git clone 后执行 docker compose up -d --build，确保部署的是 nixingmeng123 修改版，支持 grok-4.5-fast、grok-4.5-low / medium / high，以及 Free/Basic Console 视频。
```

## 常见问题

| 问题 | 说明 |
| :-- | :-- |
| 为什么 README 以前有 `jiujiu532`？ | 本仓库是从 `jiujiu532/grok2api` fork 修改而来。现在主部署说明已改成本仓库。 |
| 为什么防封版还有 `jiujiu532/privoxy-warp`？ | 那是辅助代理镜像，不是 grok2api 主程序。grok2api 主程序从本仓库 build。 |
| 为什么不能直接 `docker pull ghcr.io/jiujiu532/grok2api:latest`？ | 那会拉原作者镜像，不包含本仓库的修改。 |
| `grok-4.5-fast` 和 `grok-4.5-high` 有什么区别？ | `fast` 走 grok.com 网页端 fast；`high` 走 console.x.ai，并带 `reasoning.effort=high`。 |
| 朋友点 GitHub 链接能直接运行吗？ | 不能。GitHub 是源码仓库，朋友需要 clone 后 docker build / docker compose 部署。 |
| 图片上传 403 是什么？ | 通常是上游 asset upload 被拒或模拟上传链路不稳定，和纯文字聊天不是同一条接口。 |
| 视频停在 94% 正常吗？ | 上游可能在最后处理阶段停留较久。本版本会保持 SSE 连接并等待最多 15 分钟，不会自动重复提交同一任务。 |
| 视频返回 `imagine:content-moderated` 是什么？ | 生成结果被 xAI 内容审核拒绝。请使用合规提示词重新生成；这不是服务器或协议故障。 |

## 上游与致谢

- 修改来源：[jiujiu532/grok2api](https://github.com/jiujiu532/grok2api)
- 原始上游：[chenyme/grok2api](https://github.com/chenyme/grok2api)
- 社区：[Linux.do](https://linux.do)

## License

MIT License. 请同时尊重上游项目许可证与署名。
