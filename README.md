## how-to-pet

반려견에 대한 정보를 쉽게 찾아볼 수 있는 챗봇 🐶


`kakao-i-developers`로 만들었언 챗봇 [멍멍댕댕](https://pf.kakao.com/_sxoBAK)을 직접 코드로 구현합니다.


## Workflow

1. 입력으로 질문을 받는다.

2. 학습된 `챗봇`이 해당하는 질문에 알맞은 대답을 반환한다

  - [pytorch 의 tutorial](https://tutorials.pytorch.kr/beginner/chatbot_tutorial.html)에서의 기본 챗봇과 달리, `멍멍댕댕`은 정보습득을 위한 질문에 알맞은 대답으로 `정보를 전달`하는 데에 초점을 두어야 한다.

  - 따라서, [피자 주문 챗봇](https://www.slideshare.net/healess/python-tensorflow-ai-chatbot)을 응용한다.

### Requirements

- `pandas`

- `konlpy`

- `tensorflow`

### Reference

- [github: tensormsa_jupyter](https://github.com/TensorMSA/tensormsa_jupyter)

- [Python과 Tensorflow를 활용한 AI Chatbot 개발 및 실무 적용](https://www.slideshare.net/healess/python-tensorflow-ai-chatbot)

- [Python과 Tensorflow를 활용한 Al 챗봇 개발 3강](https://www.youtube.com/watch?v=HoT2TheIlUQ&feature=youtu.be)
