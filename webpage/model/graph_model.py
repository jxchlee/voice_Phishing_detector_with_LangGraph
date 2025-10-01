# 보이스피싱 탐지를 위한 랭그래프 모델
import os
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_tavily import TavilySearch

# .env 파일 로드
load_dotenv()




class VoicePhishingDetector:
    """보이스피싱 탐지를 위한 LangGraph 모델 클래스"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self._setup_prompts()
        self._setup_tools()
        self._build_graph()
    
    def _setup_prompts(self):
        """프롬프트 설정"""
        self.search_prompt = ChatPromptTemplate.from_messages([
            ('system', """
당신은 보이스피싱 탐지 전문가입니다.

사용자의 메시지를 분석하여, 아래 조건 중 하나라도 해당되면 '보이스피싱 탐지 툴 (voicephishing_check)'을 호출하세요:

- 돈, 송금, 계좌이체, 금융 정보, 신용카드 등의 단어가 포함됨
- 가족, 지인, 경찰, 검사 등의 단어와 함께 급박함, 협박, 회유 등의 표현이 사용됨
- 사용자가 이해할 수 없는 링크, 전화번호, 계좌번호를 요청받음
- 메시지 내용이 불분명하고 정체를 숨기려는 의도가 있음
- 기타 보이스피싱으로 의심되는 이상 징후가 보임

단, 그걸 제외한 모든 상황에서는 툴을 호출하지 마세요.
"""),
            ('human', '{context}')
        ])

        self.chat_prompt = ChatPromptTemplate.from_messages([
            ('system', """당신은 보이스피싱을 탐지하는 수사관입니다.
            텍스트로 된 전화 내용을 보고 보이스피싱 여부를 판단하십시오.
            만약 보이스피싱이라고 의심된다면 대화 내용 중 어디부분이 의심스러운 부분인지 짚으면서 설명해주세요. 검색 결과 중 관련 내용이 있다면 해당 출처와 자료를 인용해서 설명하세요.
            분석을 시작하기 전 맨 앞에서 보이스피싱 의심여부를 0 ~ 10 사이의 숫자로 표시하세요"""),
            ('human', '{context}'),
            ('ai', '''
score: [0~10 사이읜 숫자]
한 줄 평가: [진단 결과를 한 줄로 평가가]
상세 설명 및 근거: [하나씩 자세히 설명]
            ''')
        ])
    
    def _setup_tools(self):
        """도구 설정"""
        self.tavily_tool = TavilySearch(max_results=2)
        self.tools = [self.tavily_tool]
        self.llm_tool = self.search_prompt | self.llm.bind_tools(self.tools)
        self.analyze_chain = self.chat_prompt | self.llm
        self.tool_node = ToolNode(self.tools)
    
    def _build_graph(self):
        """그래프 구성"""
        # State 정의
        class State(TypedDict):
            messages: Annotated[list, add_messages]
            dummy_data: Annotated[list, add_messages]
        
        self.State = State
        
        # 그래프 빌더 생성
        graph_builder = StateGraph(State)
        
        # 노드 추가
        graph_builder.add_node("search", self._search_node)
        graph_builder.add_node("tools", self.tool_node)
        graph_builder.add_node("answer", self._answer_node)
        
        # 엣지 설정
        graph_builder.set_entry_point("search")
        graph_builder.set_finish_point("answer")
        graph_builder.add_conditional_edges("search", self._custom_tools_condition)
        graph_builder.add_edge('tools', 'answer')
        
        # 그래프 컴파일
        self.vp_search = graph_builder.compile()
    
    def _answer_node(self, state):
        """분석 노드"""
        return {
            "messages": self.analyze_chain.invoke({'context': state["messages"]}), 
            "dummy_data": 'anal step'
        }
    
    def _search_node(self, state):
        """검색 노드"""
        return {'context': [self.llm_tool.invoke(state['messages'])]}
    
    def _custom_tools_condition(self, state):
        """도구 사용 조건 판단"""
        last_message = state['messages'][-1]
        if "tool_calls" in last_message.additional_kwargs:
            return "tools"
        else:
            return "answer"
    
    def analyze_text(self, text: str) -> str:
        """텍스트 분석 메인 함수"""
        result = self.vp_search.invoke({'messages': text})
        return result["messages"][-1].content


# 외부에서 사용할 수 있는 함수
def analyze_voice_phishing(text: str) -> str:
    """보이스피싱 분석 함수"""

    detector = VoicePhishingDetector()
    return detector.analyze_text(text)


# 테스트용 코드 (직접 실행 시에만 동작)
if __name__ == "__main__":
    test_text = '''
[00:00 - 00:02]  네 고객님. 네.
[00:02 - 00:03]  번호입니다. 네.
[00:03 - 00:07]  아 근데 제가 전화해보니깐 없으시다고 하던데요?
[00:07 - 00:10]  어느 부분에 연락을 주신 건가요?
[00:10 - 00:14]  잠실이 우리은행 그 금융주차.
[00:14 - 00:17]  고객님 우리은행 지점이 여러 군데 있어서요.
[00:17 - 00:20]  주신 거 맞으신가요?
[00:20 - 00:21]  네 맞습니다.
[00:21 - 00:24]  영업 1팀으로 연락 주신 거 맞지요?
[00:24 - 00:26]  영업 1팀은 아니고 대표번호로.
[00:26 - 00:30]  아니 잠실 지점이 이게.
[00:30 - 00:37]  저희가 지점이 크다 보니까 기본적으로 이제 일종 일반 부서에서는 저희 대출팀 직원을 모를 수도 있어요.
[00:39 - 00:47]  대출 영업 대출 1팀으로 연락을 하셔서 확인해보시면 확인이 다 되시는 부분이시고요.
[00:47 - 00:49]  대출 1팀이요? 영업 대출 1팀이요?
[00:49 - 00:58]  네. 그리고 고객님 삼성카드 같은 경우에는 고객님께서 삼성카드로 연락 주시고 완납 처리해 주실 부분이시잖아요.
[00:58 - 01:00]  저희한테 뭐 해드린 게 없고.
[01:00 - 01:29]  근데 삼성카드에서 확인이 안 되니까요. 그게 제가 그러면 제가 지금. 아 제가 그 얘기 말씀 안 드렸어요. 고객님. 네. 이게 법무용 납부증명서 같은 경우는 공문서가 이미 확인이 되셨기 때문에 다른 상담원이 확인이 안 될 수가 있어요. 앞전에도 제가 분명히 말씀드렸다시피 이게 납부처리가 바로 들어가는 게 아니고 상환처리가. 네. 고객님께서 저희 쪽 대출을 못 받아보시거나. 네. 아니면은 고객님 이제 고객님께서 이제 중도에 다른 대출 진행권으로 해서 저희 쪽 부결이 나가지 않으셨을까요?
[01:30 - 01:38]  고객이 다시 환급을 받아보실 수 있도록 중간에 지금 법무팀에서 보관을 하고 예치하고 있는 상황이라서요.
[01:38 - 01:42]  그러면 제가 삼성카드에 확인을 어떻게 하죠? 전화해서?
'''
    
    result = analyze_voice_phishing(test_text)
    print(result)