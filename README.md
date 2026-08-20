# MiniMax-H3 RTX PRO 6000 部署

每张 GPU 启动一套独立的 `ComfyUI + API`，运行 NVFP4 底模、SageAttention2 和 Larry Turbo LoRA。GPU 数量由 `nvidia-smi` 自动识别；8 卡机器会注册 8 个可独立调度的实例，端口默认是 `30010`～`30017`。

默认注册信息：

- PSM：`capcut.ai_infra.federation`
- service_id：`Minimax-H3-AWS-RTX6000`
- 注册接口：`/ic/capcut/edit_gateway/v1/report_catalog`
- 每 5 秒探活并上报；未完成模型预热的实例不会上报为 alive

## 新机器安装

要求 Ubuntu 24.04、可工作的 NVIDIA 驱动、RTX PRO 6000 Blackwell、至少 100 GiB 可用磁盘，并能访问 Docker Hub、GitHub、Hugging Face 和 ReportCatalog。

```bash
git clone https://github.com/wclmxxs/rtx_6000_pro.git
cd rtx_6000_pro
./install.sh
```

这是唯一必需的安装命令。它会安装 Docker/NVIDIA Container Toolkit、下载并校验固定 revision 的模型、构建镜像、按实际 GPU 数生成 Compose、启动、逐卡预热，然后注册实例。重复执行是幂等更新。

从已经完整部署并验证过的 EC2 AMI 创建新实例时，使用快速启动模式：

```bash
./install.sh --from-ami
```

该模式仍会从 IMDS 刷新公网 IP 和 instance-id、重新生成 Compose、启动并逐卡强制预热、最后重新注册；但已有模型只检查锁定文件大小，不再读取约 47GB 数据计算 SHA256，已有 Docker 镜像也会直接复用。若模型或镜像缺失，仍会自动下载或构建。普通 `./install.sh` 保持完整哈希校验和构建流程，适合全新机器及发布验证。

如需覆盖默认值，首次运行前执行：

```bash
cp config/env.example .env
vim .env
./install.sh
```

每次运行 `install.sh` 都会优先从 AWS IMDS 重新读取公网 IPv4 和 instance-id，并覆盖 `.env` 中可能由源 AMI 遗留的 `ADVERTISE_HOST` 和 `INSTANCE_ID`；刷新前会先停止旧 Reporter，避免新实例以源实例身份继续注册。因此由已部署机器制作的 AMI 启动后，可以直接重新执行 `./install.sh`。IMDS 不可用时才使用 `.env` 中的手工配置；公网地址也不可用时安装会直接失败，不会退回私网 IP。模型文件默认放在 `/srv/minimax-h3/models`。`.env` 中的 `API_KEY` 会自动生成，仅用于节点内部健康检查和兼容的 `/v1` API。

远程素材默认允许公网地址和 `.byted.org` 内网域名；其他内网素材域名要加入 `.env` 的 `REMOTE_MEDIA_HOST_ALLOWLIST`，用逗号分隔。

## 运行接口

网关对每个已注册 lease 调用：

```text
POST /ic/capcut/edit_gateway/v2/video_generation
POST /ic/capcut/edit_gateway/v2/query/video_generation
POST /sync_infer
```

文生视频：

```bash
curl -sS -X POST http://NODE_IP:30010/ic/capcut/edit_gateway/v2/video_generation \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"MiniMax-H3",
    "content":[{"type":"text","text":"A cinematic sunrise over a lake."}],
    "resolution":"768P",
    "duration":5,
    "ratio":"16:9",
    "num_inference_steps":8
  }'
```

首尾帧：

```json
{
  "model": "MiniMax-H3",
  "content": [
    {"type": "text", "text": "The woman turns and smiles."},
    {"type": "image_url", "role": "first_frame", "image_url": {"url": "https://example.com/first.png"}},
    {"type": "image_url", "role": "last_frame", "image_url": {"url": "https://example.com/last.png"}}
  ],
  "resolution": "704P",
  "duration": 5,
  "ratio": "adaptive",
  "num_inference_steps": 8
}
```

多模态参考把 role 改为 `reference_image`、`reference_video` 或 `reference_audio`，对应类型为 `image_url`、`video_url`、`audio_url`。首尾帧不能和参考媒体混用。所有任务至少包含一条非空 text。纯文本任务必须给非 `adaptive` 的 ratio。

查询：

```bash
curl -sS -X POST http://NODE_IP:30010/ic/capcut/edit_gateway/v2/query/video_generation \
  -H 'Content-Type: application/json' \
  -d '{"model":"MiniMax-H3","task_id":"TASK_ID"}'
```

状态为 `queued`、`running`、`succeeded`、`failed` 或 `expired`。成功时 `task.content.url` 是节点上的 MP4 下载地址；失败和所有业务接口 HTTP 错误都会返回明确的 `error.type`、`error.message` 和 `error.http_code`。

任务完成或失败后，会立即删除该任务下载的输入素材和 ComfyUI 原始产物；对外提供的最终 MP4 默认保留 12 小时。后台每 60 秒清理一次，到期任务变为 `expired`，内容接口返回 HTTP 410。可通过 `.env` 的 `OUTPUT_TTL_SECONDS` 和 `CLEANUP_INTERVAL_SECONDS` 调整。

接口参数：resolution 仅支持 `768P`/`704P`，duration 为 4～15，ratio 支持 `adaptive`、`21:9`、`16:9`、`4:3`、`1:1`、`3:4`、`9:16`。`num_inference_steps` 按真实 NFE 计数；4/6/8 NFE 的 T2V/FL2V 自动使用 Turbo LoRA，其他步数走底模采样器，REF2V 始终不使用该 LoRA。

## 运维

```bash
./status.sh
./smoke_test.sh
sudo docker compose --env-file .env -f .generated/compose.yaml logs -f h3-reporter
```

发布或模型版本变化时修改 `RELEASE_ID` 后重跑 `./install.sh`，会重新逐卡预热后再上报。实例是单卡 worker，单实例 concurrency 应配置为 1；网关在同一 service_id 下调度所有机器的全部实例。

注意：当前成功 URL 指向产出该视频的节点本地文件。若业务要求跨实例长期保存，应由网关下载后落 TOS，或在本服务前增加统一对象存储上传步骤；不能在任务完成后立即回收节点本地输出。
