import konlpy
from konlpy.tag import Mecab
import pandas as pd

''' for test Mecab

mecab = Mecab('')
print(mecab.pos('아버지가방에들어가신다'))
'''

## 1. slot 구성
input_data = "생후 4개월인데 어떤 주사 맞춰야 해?" #채팅에서 입력

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

#향후 의도 구분을 위한 학습Data로 사용
intent_list = {
                "먹이" : ["먹이","먹어도","안될까"],
                "접종" : ["예방접종","주사", "접종", "예방", "백신"],
                "질병" : ["정보", "알려", "질병", "아파", "증상", "처방"],
                "훈육" : ['놀이', '훈육', '방법', '어떻게'],
                "특성" : ['성격', '특징', '특성']
                } 
#각 의도별 Slot 구성
story_slot_entity = {"먹이": {"음식" : None},
                     "접종": {"종류" : None, "시기" : None},
                     "질병": {"종류" : None, '견종': None, "증상": None},
                     "훈육": {"종류": None}, 
                     "특성": {'견종': None}                   
                    }

## 3. 형태소 분석 

mecab = Mecab('/usr/local/lib/mecab/dic/mecab-ko-dic') #고유명사는 별도 NNP 등록
preprocessed = mecab.pos(request.get('input_data'))
print(preprocessed)

# [('생후', 'NNG'), ('4', 'SN'), ('개월', 'NNBC'), ('인데', 'VCP+EC'), ('어떤', 'MM'), ('주사', 'NNG'), ('맞춰야', 'VV+EC'), ('해', 'VX+EF'), ('?', 'SF')]

'''

input_data2 = ""
preprocessed2 = mecab.pos(request.get('input_data2'))
print(preprocessed2)

# [('말', 'NNG'), ('티즈', 'NNP'), ('성격', 'NNG'), ('이', 'JKS'), ('어때', 'VA+EF'), ('?', 'SF')]
# '말티즈' 등 견종을 하나의 고유명사로 인식해야 함 

input_data3 = "초콜릿 먹여도 돼?"
preprocessed3 = mecab.pos(request.get('input_data3'))
print(preprocessed3)

# [('초콜릿', 'NNG'), ('먹여', 'VV+EC'), ('도', 'JX'), ('돼', 'VV+EF'), ('?', 'SF')]

input_data4 = "어떤 주사 맞춰야 해?" #채팅에서 입력
[('어떤', 'MM'), ('주사', 'NNG'), ('맞춰야', 'VV+EC'), ('해', 'VX+EF'), ('?', 'SF')]

'''

## 4. Intent 도출(Rule Based)

intent_id ="접종"
slot_value = story_slot_entity.get("접종")
'''접종(종류, 시기), 먹이(음식), 질병(종류, 견종, 증상), 훈육(종류), 특성(견종) '''


## 5. NER 도출(Rule Based)

# 접종 
inject_list = ['종합백신','DHPPL','코로나', '켄넬코프', 'Kennel Cough','감기', '광견병', '인플루엔자', '주사', '어떤'] # 접종 종류
when_list = ['1년', '매년', '매달', '생후', '개월', '보강접종', '재접종', '주', '언제', '몇살때', '몇살', '나이', '언제'] # 시기 

# 먹이
eat_list = ['사과','배','아보카도','감','멜론','블루베리','바나나','딸기','수박',
        '파슬리','배추','상추','감자','토마토','당근', '무','브로콜리', '단호박', '고구마', '감자', '오이', '양배추', 
        '익힌연어,북어,계란노른자,두부,플레인요거트', '오리고기,닭고기,양고기,소고기,칠면조고기', #먹어도 됨 
        '포도', '건포도', '자몽', '무화과', '자두', '건자두', '양파', '파' , '마늘', '고추', '버섯', '날계란', '견과류', 
        '오징어, 쥐포, 생선, 전복, 소라, 김', '초콜릿, 껌, 카라멜, 과자, 빵, 우유, 탄산음료', 
        '술','짠','닭뼈' ,'생선뼈','커피','홍차','녹차','기름진거', '고양이밥' # 먹으면 안됨 
        ] # 음식

# 질병 
dis_list = ['질병','조심',''] # 질병
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
teach_list = ['앉', '이리와', '기다려', '엎드려', '빵', '가슴줄', '착용', '놀이', '간식', '안아', '친구'] # 훈육 종류

# 특성 - 견종 
feat_list = ['성격', '특징', '특성', '특이']# 특성 


## 6. Dictionary 기반 Slot 구성

#전처리 로직으로 문장내 연관 단어만 도출(조사, 마침표제거등)
for pos_tag in preprocessed:
    if (pos_tag[1] in ['NNG', 'NNP','SL','MAG']): #명사, 영어만 사용
        if pos_tag[0] in inject_list:  # 접종 종류 List 에서 검색
            slot_value["종류"] = pos_tag[0] 
        elif pos_tag[0] in when_list: # 시기 List 에서 검색
            slot_value["시기"] = pos_tag[0] 
print (story_slot_entity.get('접종'))
# {'종류': '주사', '시기': '생후'}


## 7. 빈 Slot 검색

if(None in slot_value.values()): #빈 Slot 출력
    key_values = ""
    for key in slot_value.keys():
        if(slot_value[key] is None):
            key_values = key_values + key + ","
    output_data = key_values + '말씀해주세요'
else:
    output_data = "대답해드릴게용."
            
print (output_data)

response = {
            "intent_id" : "",
            "input_data":input_data, 
            "request_type" : "text",
            "story_slot_entity" : {},
            "output_data" : ""
          }
response["output_data"]= output_data

print (response['output_data'])