# summary_and_classify_remopoke
リモートポケモン学会の発表を要約して、それを分類するモデルを作る

## Usage

### Setup

```bash
poetry install
```

And get OpenAI API Key.

### Get Summary

```bash
OPENAI_API_KEY=YOUR_API_KEY make get_presentation_summary
```

## Summary Example

- video: https://www.youtube.com/watch?v=HV4NYf2fDHE
- title: 超人と携帯獣を繋ぐモノ【リモポケ学会#mini2】

summary:
この動画は、ポケモンとロボットアニメの関連性についてのプレゼンテーションを特集しています。プレゼンターは、自身がアマチュアの小説家であり、リモートポケモン学会のスタッフでもあることを紹介します。彼はまた、ポケモンが1996年に始まり、25年以上にわたりテレビゲーム、アニメ、漫画、大規模なイベントなど、多岐にわたる活動を展開してきたことを説明します。

彼は、ポケモンがロボットアニメに影響を受けたことを示すために、特定のロボットアニメを引用します。彼は、ポケモンの開発を担当しているゲームフリークが、マニア（フリーク）が参加していると述べ、特撮作品やロボットアニメに精通していることで、ポケモンをより深く楽しむことができると主張します。

彼は視聴者に、特定のロボットアニメを認識できるかどうかを問いかけますが、その数が多すぎて難しいと認識しています。彼は、ロボットアニメが約800作品もあるとされているため、どの作品を見るべきかを決めるのは難しいと述べます。

その解決策として、彼はスーパーロボット大戦シリーズを推奨します。これは、多くのロボットアニメがクロスオーバーしているシリーズで、ロボットアニメのカタログとも言えます。彼は、このシリーズを遊び、気に入ったロボットが出演している作品を見ることを提案します。そして、その作品からポケモンとの共通点を見つけ、開発に対する理解を深めることが、次世代型のポケモンの楽しみ方になると述べています。

## Classification Example

TBD
