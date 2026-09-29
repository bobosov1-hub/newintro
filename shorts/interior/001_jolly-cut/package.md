# 홈 인테리어 쇼츠 #001 — 졸리컷

- 제목 후보
  1. 욕실 모서리에 이게 없으면, 돈 더 쓴 집입니다
  2. 인테리어 업자가 "졸리컷 하실래요?" 물어보면
  3. 액자 만드는 법으로 욕실을 마감하면 생기는 일
- 길이: 약 45초
- TTS 속도: 1.2~1.3배
- 핵심 비유: **액자 틀 모서리** (두 나무를 비스듬히 잘라 맞붙이는 것 = 졸리컷)
- 반복 재생 장치: 마지막 문장이 첫 문장으로 이어짐

---

## 1. TTS 대본 (그대로 넣으면 됩니다)

```
욕실 모서리에. 이런 쇠 막대. 보이시죠.
이게 없는 집은. 돈을. 더 쓴 집일 수 있습니다.

이 막대 이름은. 코너비드.
타일 두 장이 만나는 모서리를. 덮어 주는. 뚜껑입니다.

그런데. 이 뚜껑 없이. 모서리를 끝내는 방법이 있습니다.
바로. 졸리컷.

액자 틀 모서리. 보신 적 있으시죠.
나무 두 개를. 비스듬하게 잘라서. 딱 맞붙입니다.
졸리컷이. 바로 그겁니다.
타일 끝을. 사십오 도로 갈아서. 모서리에서 맞붙이는 거죠.
그래서 이음매가. 선 하나로만 보입니다.
마치. 돌덩어리 하나를. 통째로 깎은 것처럼요.

대신. 대가가 있습니다.
첫째. 돈.
타일을 한 장씩 갈아야 해서. 시공비가. 확 올라갑니다.
둘째. 깨짐.
얇게 갈린 모서리는. 세게 부딪히면. 이가 나갈 수 있습니다.
가공하다 깨지는 타일도 많아서. 타일도. 넉넉히 사야 하고요.
셋째. 시간.
모서리마다. 갈고. 맞추고. 그만큼. 공사가 길어집니다.

그래서. 이렇게 나눠 쓰는 집이 많습니다.
눈에 제일 잘 보이는 곳은. 졸리컷.
손이 자주 닿는 곳은. 코너비드.

이제. 남의 집 욕실에 가면.
모서리부터. 보게 되실 겁니다.
욕실 모서리에. 이런 쇠 막대.
```

> 마지막 줄 "욕실 모서리에. 이런 쇠 막대."가 첫 줄로 그대로 이어지게 편집합니다 (끊김 없이 반복 재생).

---

## 2. 장면 구성 + 구글 플로우 프롬프트

공통 설정: **세로 9:16**, 장면당 약 8초, 플로우의 자체 음성·음악은 끄거나 편집에서 제거.
공통 스타일 문구(모든 프롬프트 끝에 붙임):
`photorealistic 3D architectural visualization, clean modern Korean apartment bathroom, soft daylight, slow cinematic camera, shallow depth of field, no text, no people, vertical 9:16`

| # | 대본 구간 | 화면 | 플로우 프롬프트 (영문) |
|---|---|---|---|
| S1 | 욕실 모서리에… 더 쓴 집일 수 있습니다 | 욕실 벽 바깥 모서리의 금속 코너비드로 빠르게 줌인 | `Fast push-in camera toward the outer corner of a tiled bathroom wall, a thin silver metal corner bead strip runs vertically along the edge, gray large-format porcelain tiles` |
| S2 | 이 막대 이름은 코너비드… 뚜껑입니다 | 코너비드를 단면으로 보여줌. 타일 두 장이 직각으로 만나고 막대가 덮음 | `Cross-section cutaway view of two porcelain tiles meeting at a 90 degree outer corner, a metal L-shaped trim strip covering the joint, rotating slowly, studio lighting, white background` |
| S3 | 그런데 이 뚜껑 없이… 바로. 졸리컷. | 코너비드가 위로 쓱 빠지고, 이음매 없는 날카로운 모서리가 드러남 | `The metal corner trim slides upward and disappears, revealing a seamless sharp tiled corner where two tiles meet perfectly with a hairline joint, dramatic reveal lighting` |
| S4 | 액자 틀 모서리… 바로 그겁니다 | 나무 액자 틀 두 조각이 45도로 잘려 맞붙는 장면 | `Two wooden picture frame pieces with 45 degree mitered ends slide together and join perfectly at the corner, macro shot, warm light, clean workbench` |
| S5 | 타일 끝을 사십오 도로 갈아서… 선 하나로만 | 그라인더로 타일 모서리를 비스듬하게 가는 클로즈업 → 두 장이 맞붙음 | `Close-up of a wet tile grinder beveling the edge of a porcelain tile at 45 degrees, fine water spray, then two beveled tiles join at an outer corner forming a clean single line` |
| S6 | 마치 돌덩어리 하나를 통째로 깎은 것처럼요 | 완성된 욕실 모서리를 천천히 돌아가는 고급스러운 샷 | `Slow orbit around a luxurious bathroom wall corner finished with mitered porcelain tiles, looks like a single carved stone block, marble texture, elegant lighting` |
| S7 | 첫째. 돈… 확 올라갑니다 | 타일 더미 위로 비용이 쌓이는 느낌 (동전/지폐는 편집에서 그래픽으로) | `A stack of porcelain tiles on a construction site floor, each tile being picked up one by one, time-lapse feel, dusty warm light` |
| S8 | 둘째. 깨짐… 넉넉히 사야 하고요 | 얇게 갈린 모서리가 톡 부딪혀 작은 조각이 튀는 슬로모션 | `Extreme slow motion of a thin beveled tile corner being bumped by a small object, a tiny chip breaking off, macro lens, dramatic side light` |
| S9 | 셋째. 시간… 공사가 길어집니다 | 창밖 빛이 낮→밤으로 바뀌는 타임랩스, 공사 중인 욕실 | `Time-lapse of an unfinished bathroom renovation, daylight through the window shifts from morning to night, tools and tile pieces on the floor` |
| S10 | 그래서 이렇게 나눠 쓰는 집이… 코너비드 | 화면 분할 느낌: 창가 쪽 모서리는 매끈, 출입구 모서리는 코너비드 | `Wide shot of a modern bathroom showing two outer wall corners, the corner near the vanity has a seamless mitered tile edge, the corner near the doorway has a thin metal corner bead, even lighting` |
| S11 | 이제 남의 집 욕실에 가면… 쇠 막대 | S1과 같은 구도로 끝나서 첫 장면에 이어짐 | S1 프롬프트 재사용 (시작 구도를 S1 첫 프레임과 맞춤) |

---

## 3. 편집 그래픽 (AI 영상에 맡기지 않고 편집에서 얹음)

| 장면 | 그래픽 |
|---|---|
| S1 | 빨간 원으로 코너비드 표시 + 텍스트 `코너비드` |
| S4~S5 | 빨간 치수선 + 각도 표시 `45°` (두 조각 각각) |
| S5 끝 | 이음매를 따라 빨간 얇은 선 한 줄 |
| S7 | `시공비 ↑` 배지가 튀어나옴 |
| S8 | 깨지는 순간 빨간 테두리 깜빡임 |
| S10 | 왼쪽 `졸리컷` / 오른쪽 `코너비드` 라벨 |

---

## 4. 효과음 위치

| 타이밍 | 효과음 |
|---|---|
| 첫 단어와 동시 | 짧은 "팅" 금속음 (코너비드 느낌) |
| "바로. 졸리컷." | 휙 빠지는 소리(swoosh) + 낮은 "둥" 임팩트 |
| 액자 조각이 맞붙는 순간 | 딸깍 (나무 맞물림) |
| 그라인더 장면 | 짧은 그라인더 소리 (0.5초 이내, 대사 가리지 않게) |
| "첫째 / 둘째 / 셋째" | 매번 같은 짧은 틱 소리 (리듬감) |
| 타일 깨지는 순간 | 작은 "짹" 파손음 + 0.3초 무음 |
| 타임랩스 | 시계 째깍 3회 |
| 마지막 줄 → 처음 | 효과음 없이 바로 이어 붙임 (반복 재생이 티 안 나게) |

배경음: 잔잔한 로파이 비트, 대사 구간 -20dB 이하.

---

## 5. 사실 확인 메모 (검수용)

- 졸리컷 = 타일 모서리를 45도(이상)로 연마해 바깥 모서리에서 맞붙이는 마감. 코너비드보다 깔끔하지만 가공비와 파손(로스)이 늘어남 — [고강타일 블로그](https://inblog.ai/gogangtile/pros-and-cons-of-jolly-trim-porcelain-tile)
- 시공비 "1.5~2배", "한 칸당 약 40만 원 추가"는 커뮤니티 글 제목·검색 요약뿐이고 원문 확인 불가(403). 고강타일·LX Z:IN 글에도 수치 없음 → **대본에서 숫자 뺌** ("돈을 더 쓴", "확 올라갑니다") — [뽐뿌 글](https://m.ppomppu.co.kr/new/bbs_view.php?id=interior&no=13532), [LX Z:IN](https://www.lxzin.com/styling/style-guide/detail/1201)
- **시공 기간은 구체적 수치 출처를 찾지 못함** → 대본에는 숫자 없이 "길어진다"로만 표현함
- "보이는 곳은 졸리컷, 손 닿는 곳은 코너비드"는 파손 위험에서 나온 일반적인 절충안 표현. 단정하지 않도록 "나눠 쓰는 집이 많다"로 씀
