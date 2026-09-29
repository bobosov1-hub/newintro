# 세로 쇼츠 자동 편집기

장면 영상(구글 플로우 등) + `episode.json`(대본·그래픽·효과음) → 1080×1920 완성본 mp4.

```bash
pip install -r shorts/requirements.txt

# 1) 화면만 먼저 확인 (음성 없이)
python shorts/make_short.py shorts/interior/001_jolly-cut/episode.json --clips "장면폴더" --tts silent

# 2) Voicebox 목소리로 완성 (Voicebox 앱을 켜 둔 상태)
python shorts/make_short.py shorts/interior/001_jolly-cut/episode.json --clips "장면폴더" --tts voicebox
#    → 처음엔 목소리 목록이 나옵니다. 쓸 id를 episode.json의 voicebox.profile_id에 넣거나
#      --voicebox-profile <id> 로 지정하세요.
```

| 옵션 | 뜻 |
|---|---|
| `--tts voicebox` | PC의 Voicebox 앱(`http://127.0.0.1:17493`)으로 줄마다 음성 생성 |
| `--tts files --voice-dir 폴더` | 직접 만든 음성 파일 사용 (`L000.wav`, `L001.wav` … 대본 줄 순서) |
| `--tts edge` | 마이크로소프트 무료 음성 (인터넷 필요) |
| `--tts silent` | 무음, 화면·그래픽 위치 확인용 |
| `--order mtime` | 장면 파일 이름이 S1…이 아닐 때 다운로드 순서로 짝짓기 |
| `--sfx-dir 폴더` | `ting.wav`, `swoosh.wav` 등 진짜 효과음이 있으면 합성음 대신 사용 |
| `--bgm 파일` | 배경음악 |

`episode.json`의 `speed`는 Voicebox·files 음성에 적용되는 배속입니다 (edge는 `rate`).
음성은 `output/_tts_cache`에 저장되어, 대사를 안 바꾼 줄은 다시 만들지 않습니다.
