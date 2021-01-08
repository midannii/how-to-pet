import konlpy
from konlpy.tag import Mecab
import pandas as pd

''' for test Mecab

mecab = Mecab('')
print(mecab.pos('아버지가방에들어가신다'))
'''

## 1. slot 구성
input_data = "강아지를 키우기로 했는데, 혹시 예방접종이 필요할까요?" #채팅에서 입력

#출력값
output_data = ""
#챗봇에서 서버와 주고받을 API format
request = {
            "intent_id" : "",
            "input_data":input_data,
            "request_type" : "text",
            "story_slot_entity" : {},
            "output_data" : output_data
          }

## 2. 기본 DB 설정

#향후 의도 구분을 위한 학습 Data로 사용
intent_list = { # 모두 정보 요청이긴 하지만, 목적에 따라 종류 나눔 - 먹이질문, 훈련법질문, 질병질문, 성격(견종별)질문 등
                "먹이" : ["줘도", "먹여도", "먹이", "밥", "사료"],
                "훈련" : ["훈육", "훈련", "어떻게", "놀이", "놀아", "교육", "가르치기"],
                "질병" : ["병원", "질병", "이상", "아파", "조심", "병", "예방"],
                "성격": ['특징', '성격', '어때'],
                "접종": [ "예방", "접종", "주사", "언제", "맞혀"],
                "진단": ["병원", "아파", "이상", "병"]
                }
#각 의도별 Slot 구성
story_slot_entity = {"먹이": {"먹이종류" : None},
                     "훈련": {"상황" : None},
                     "질병": {"견종" : None}, # 주의할 질병 정보 소개
                     "접종": {"나이": None}, # 예방접종 안내
                     "진단": {"견종": None, "증상": None}, # 증상으로 의심 질병 제시
                     "성격": {"견종": None}
                    }

## 3. 형태소 분석

mecab = Mecab('/usr/local/lib/mecab/dic/mecab-ko-dic') #고유명사는 별도 NNP 등록
preprocessed = mecab.pos(request.get('input_data'))
print(preprocessed)

# [('강아지', 'NNG'), ('를', 'JKO'), ('키우', 'VV'), ('기', 'ETN'), ('로', 'JKB'), ('했', 'VV+EP'), ('는데', 'EC'), (',', 'SC'), ('혹시', 'MAG'), ('예방', 'NNG'), ('접종', 'NNG'), ('이', 'JKS'), ('필요', 'NNG'), ('할까요', 'XSA+EF'), ('?', 'SF')]


## 4. Intent 도출(Rule Based)

intent_id = "접종"
slot_value = story_slot_entity.get(intent_id)

## 5. NER 도출(Rule Based)

# 먹이 - 먹이종류
feed_list = ['사과','배','아보카도','감','멜론','블루베리','바나나','딸기','수박',
        '파슬리','배추','상추','감자','토마토','당근', '무','브로콜리', '단호박', '고구마', '감자', '오이', '양배추',
        '익힌연어,북어,계란노른자,두부,플레인요거트', '오리고기,닭고기,양고기,소고기,칠면조고기', #먹어도 됨
        '포도', '건포도', '자몽', '무화과', '자두', '건자두', '양파', '파' , '마늘', '고추', '버섯', '날계란', '견과류',
        '오징어, 쥐포, 생선, 전복, 소라, 김', '초콜릿, 껌, 카라멜, 과자, 빵, 우유, 탄산음료',
        '술','짠','닭뼈' ,'생선뼈','커피','홍차','녹차','기름진거', '고양이밥' # 먹으면 안됨
        ] # 음식

# 질병
symt_list = ['식욕', '구토', '설사', '복부팽창', '산만',
            '거친 숨소리','기침', '콧물',
            '발작', '경련', '발 헛디딤','헛디딤','헛디', '기절', '빙글빙글',
            '털빠짐', '발진', '체중변화', '체중','염증',
            '무기력', '과민반응', '불안', '회피', '대소변', '고열', '열'] # 증상

# 견종
breed_list = ['치와와, 말티즈, 미니어처 핀셔, 파피용, 포메라니안, 푸들, 시추',
            '미니어처 슈나우저, 요크셔테리어', '비글, 닥스훈트, 아프간 하운드',
            '코커스패니얼', '골든 리트리버', '시베리안 허스키', '알래스칸 말라뮤트', '도베르만 핀셔',
            '보더콜리', '웰시코기', '저먼 셰퍼드', '진돗개', '삽살개', '풍산개', '불독', '시바견', '달마시안',
            '비숑프리제', '비숑', '테리어', '하운드', '리트리버', '허스키', '오스트레일리언 테리어', '비글', '사모예드']

# 훈육
teach_list = ['앉', '이리와', '기다려', '엎드려', '빵', '가슴줄', '착용', '놀이', '간식', '안아', '친구'] # 훈육 상황(종류)

# 접종
inject_list = ['종합백신','DHPPL','코로나', '켄넬코프', 'Kennel Cough','감기', '광견병', '인플루엔자', '주사', '어떤'] # 예방접종 종류
age_list = ['1년', '매년', '매달', '생후', '개월', '살'] # 나이


## 6. Dictionary 기반 Slot 구성

#전처리 로직으로 문장내 연관 단어만 도출(조사, 마침표제거등)
ambiguous = []
for pos_tag in preprocessed:
    if (pos_tag[1] in ['NNG', 'NNP','SL','MAG']): #명사, 영어만 사용
        if pos_tag[0] in feed_list: #먹이 List 에서 검색
            slot_value["먹이"] = pos_tag[0]
        elif pos_tag[0] in teach_list: #훈육 상황(종류) List 에서 검색
            slot_value["훈련"] = pos_tag[0]
        elif pos_tag[0] in breed_list: #견종 List 에서 검색
            slot_value["질병"] = pos_tag[0]
            slot_value["진단"] = pos_tag[0]
            slot_value["성격"] = pos_tag[0]
        elif pos_tag[0] in age_list: #나이 List 에서 검색
            slot_value["접종"] = pos_tag[0]
        elif pos_tag[0] in symt_list: #증상 List 에서 검색
            slot_value["진단"] = pos_tag[0]
        else: ambiguous.append(pos_tag[0])
        #print(pos_tag[0])

## 7. 빈 Slot 검색

if(None in slot_value.values()): #빈 Slot 출력
    key_values = ""
    for key in slot_value.keys():
        if(slot_value[key] is None):
            key_values = key_values + key + ","
    output_data = key_values + ' 선택해주세요'
else:
    output_data = "정보 입력이 완료되었습니다"
print (output_data)

# 나이, 선택해주세요

response = {
            "intent_id" : "",
            "input_data":input_data,
            "request_type" : "text",
            "story_slot_entity" : {},
            "output_data" : ""
          }
response["output_data"]= output_data

print (response['output_data'])
