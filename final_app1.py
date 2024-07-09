import streamlit as st
from streamlit_option_menu import option_menu
import openai
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OllamaEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOllama
from langchain.callbacks.base import BaseCallbackHandler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import io

# Set OpenAI API key
openai.api_key = 'your openai api key'

def initialize_session_state():
    """Initialize session state variables."""
    for key in ['confirmed', 'step1_done', 'step2_done', 'step3_done', 'step4_done', 'current_step']:
        if key not in st.session_state:
            st.session_state[key] = False
    if 'chat_histories' not in st.session_state:
        st.session_state.chat_histories = {}

def generate_main_response(prompt):
    """Generate a response from OpenAI API."""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=3000
    )
    return response.choices[0].message['content'].strip()


def generate_response(chat_history):
    """Generate a response from OpenAI API using chat history."""
    messages = [{"role": "system", "content": "You are a helpful assistant."}] + chat_history
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=3000
    )
    return response.choices[0].message['content'].strip()

def generate_model_steps(modeling_direction):
    """Generate modeling steps based on the user's direction."""
    prompt = (
        f"유저가 원하는 모델링 방향은 다음과 같습니다: {modeling_direction}. "
        "이를 기반으로 모델링 절차를 \nStep 1: 데이터 시각화, \nStep 2: 데이터 전처리, "
        "Step 3: 모델링 및 모델 성능 평가(최소 5개의 모델 사용), \nStep 4: 예측 결과 분석 및 모델 개선, "
        "Step 5: 레포트 생성"
    )
    response = generate_main_response(prompt)
    return response

def display_steps():
    """Display suggested modeling steps."""
    st.write("제안된 모델링 절차:")
    for step_key, step_desc in st.session_state.steps_dict.items():
        st.write(f"{step_key}: {step_desc}")

def handle_step_completion(step_key):
    """Handle the completion of each step."""
    if st.button(f"Finalize {step_key}"):
        st.session_state[f"{step_key}_done"] = True
        st.experimental_rerun()

def chat_ui(step_key):
    """LLM 기반 채팅 UI."""
    if step_key not in st.session_state.chat_histories:
        st.session_state.chat_histories[step_key] = []

    with st.form(key='chat_form', clear_on_submit=True):
        user_input = st.text_input("You: ")
        submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            # Add user's message to chat history
            st.session_state.chat_histories[step_key].append({"role": "user", "content": user_input})

            # Generate response using LLM with the entire chat history
            response = generate_response(st.session_state.chat_histories[step_key])
            st.session_state.chat_histories[step_key].append({"role": "assistant", "content": response})

    # Display chat history
    for message in st.session_state.chat_histories[step_key]:
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown(f"**Assistant:** {message['content']}")

def main_page():
    """Main page content."""
    st.title('노코드 머신러닝 모델링 툴 Dr.AUTOGEN 1.0')
    st.write("""
    AI를 만드는 AI Autogen이 자동으로 데이터 예측 모델을 만들고 그 과정을 함께 합니다.
    \n모델 방향성에는 모델링의 목적 및 맥락을 적어주세요. 예측에 사용할 변수(X 변수)와 예측해볼 변수(Y)를 미리 설정해 주시면 도움이 됩니다.
    \n예: 내가 업로드한 파일의 1, 2번째 열을 제외한 나머지 열들을 통해서 1, 2번째 열을 예측해줘
    """)
    st.markdown("---")
    st.markdown("### 모델에서 사용할 데이터 파일을 업로드해주세요")
    uploaded_file = st.file_uploader("파일을 여기로 드래그 앤 드롭하세요", type=["csv", "xlsx"])
 
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            st.session_state.dataframe = df
            st.session_state.uploaded_file = uploaded_file
            modeling_direction = st.text_area("모델링 방향을 입력하세요:")
            if st.button('모델링 세부 Task 생성'):
                with st.spinner("AUTOGEN이 답변 생성 중"):
                     time.sleep(5)
                steps_response = generate_model_steps(modeling_direction)
                steps_list = steps_response.split('\n')
                steps_dict = {}
                step_key = ""
                for step in steps_list:
                    if step.startswith("Step"):
                        step_key = step.split(':')[0].strip()
                        steps_dict[step_key] = step.split(':')[1].strip()
                    else:
                        if step_key:
                            steps_dict[step_key] += ' ' + step.strip()
 
                st.session_state.steps_dict = steps_dict
                st.session_state.current_step = 0
                st.session_state.results = {}
                st.experimental_rerun()
        except Exception as e:
            st.error(f"파일을 처리하는 중 오류가 발생했습니다: {e}")
 
    if 'steps_dict' in st.session_state and not st.session_state.get('ok_clicked', False):
        display_steps()
    st.markdown("---")
    st.markdown("### AUTOGEN이 제시한 모델링 방향성을 확인하시고 Confirm을 눌러주세요.") 
        
    if st.button("Confirm"):
        st.session_state.confirmed = True
        st.experimental_rerun()
    st.markdown("---")
    st.markdown("### 버튼을 누른 뒤 세부 모델링 탭으로 이동해주세요")    



def detail_modeling_page(sub_selected):
    """Detail modeling page content."""
    st.title("Detail Modeling")

    step_keys = ["Step 1", "Step 2", "Step 3", "Step 4"]
    for step_key in step_keys:
        if sub_selected and sub_selected.startswith(step_key):
            st.header(step_key)
            step_desc = st.session_state.steps_dict.get(step_key, '')
            for line in step_desc.split('. '):
                st.write(line)

            # LLM 기반 채팅 UI 추가
            st.markdown("---")
            st.markdown("### AUTOGEN에게 모델링 가이드를 전달해주세요")
            chat_ui(step_key)
            
            
            # 데이터프레임이 업로드되었는지 확인
            if st.session_state.dataframe is None:
                st.error("업로드된 파일이 없습니다. 메인 페이지에서 파일을 업로드해주세요.")
                return
 
            df = st.session_state.dataframe
 
            # Step 1: Visualization 버튼 추가
            if step_key == "Step 1":
                st.markdown("---")
                st.markdown("### 모델링 진행 버튼을 누르면 모델링이 시작됩니다")
                if st.button("모델링 진행"):
                    try:
                        with st.spinner("AUTOGEN이 답변 생성 중"):
                            time.sleep(3)
                        # Describe 데이터 저장
                        df_describe = df.describe()
                        st.session_state.df_describe = df_describe
                        st.write("### DataFrame Describe")
                        st.write(df_describe)

                        st.write("### Box Plot")
                        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
                        for i in range(0, len(num_cols), 10):
                            cols_to_plot = num_cols[i:i+10]
                            st.write(f"#### Box Plot: Columns {i+1} to {i+10}")
                            fig, ax = plt.subplots(figsize=(12, 6))
                            sns.boxplot(data=df[cols_to_plot], ax=ax)
                            st.pyplot(fig)
                            
                            # Box plot 이미지 세션에 저장
                            buf = io.BytesIO()
                            fig.savefig(buf, format="png")
                            buf.seek(0)
                            st.session_state[f'box_plot_{i}'] = buf

                    except Exception as e:
                        st.error(f"데이터를 시각화하는 중 오류가 발생했습니다: {e}")

            # Step 2: Data Preprocessing 버튼 추가
            if step_key == "Step 2":
                st.markdown("---")
                st.markdown("### 모델링 진행 버튼을 누르면 모델링이 시작됩니다")
                if st.button("모델링 진행"):
                    try:
                        with st.spinner("AUTOGEN이 답변 생성 중"):
                            time.sleep(3)
                        # 결측치 제거
                        st.write("### 결측치 제거")
                        missing_values = df.isnull().sum().sum()
                        st.session_state.missing_values = missing_values
                        st.write(f"총 결측치 개수: {missing_values}")
                        df = df.dropna()
                        st.write(f"결측치 제거 후 데이터 크기: {df.shape}")

                        # 이상치 제거 (IQR 방식)
                        st.write("### 이상치 제거 (IQR 방식)")
                        Q1 = df.quantile(0.25)
                        Q3 = df.quantile(0.75)
                        IQR = Q3 - Q1
                        outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)
                        num_outliers = outliers.sum()
                        st.session_state.num_outliers = num_outliers
                        st.write(f"총 이상치 개수: {num_outliers}")
                        df = df[~outliers]
                        st.write(f"이상치 제거 후 데이터 크기: {df.shape}")
                        # Scaling
                        st.write("### 데이터 스케일링")
                        min_max_scaler = MinMaxScaler()
                        standard_scaler = StandardScaler()
 
                        df_min_max_scaled = pd.DataFrame(min_max_scaler.fit_transform(df), columns=df.columns)
                        df_standard_scaled = pd.DataFrame(standard_scaler.fit_transform(df), columns=df.columns)
 
                        # 세션에 저장
                        st.session_state.min_max_scaled = df_min_max_scaled
                        st.session_state.standard_scaled = df_standard_scaled
 
                        st.write("MinMax Scaler 데이터 셋:")
                        st.write(df_min_max_scaled.head())
 
                        st.write("Standard Scaler 데이터 셋:")
                        st.write(df_standard_scaled.head())



                    except Exception as e:
                        st.error(f"데이터를 전처리하는 중 오류가 발생했습니다: {e}")
 
# Step 3: Model Selection and Tuning 버튼 추가
            if step_key == "Step 3":
                st.markdown("---")
                st.markdown("### 모델링 진행 버튼을 누르면 모델링이 시작됩니다")
                if st.button("모델링 진행"):
                    try:
                        with st.spinner("AUTOGEN이 답변 생성 중"):
                            time.sleep(3)
                        st.write("### 모델 선택 및 튜닝")

                        # 모델 리스트
                        models = {
                            'Decision Tree': DecisionTreeRegressor(),
                            'Random Forest': RandomForestRegressor(),
                            'Gradient Boosting': GradientBoostingRegressor(),
                            'Support Vector Machine': SVR()
                        }

                        # 튜닝 파라미터
                        param_grid = {
                            'Decision Tree': {'max_depth': [3, 5, 7]},
                            'Random Forest': {'n_estimators': [50, 100, 150]},
                            'Gradient Boosting': {'n_estimators': [50, 100, 150]},
                            'Support Vector Machine': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
                        }

                        # 각 데이터 셋에 대해 모델을 fit 및 튜닝
                        results = []
                        for scale_type, scaled_df in [('MinMax Scaled', st.session_state.min_max_scaled), ('Standard Scaled', st.session_state.standard_scaled)]:
                            X = scaled_df.drop(columns=['LAB_612_HN_DIST_NIR_EP'])
                            y = scaled_df['LAB_612_HN_DIST_NIR_EP']
                            for model_name, model in models.items():
                                grid_search = GridSearchCV(model, param_grid[model_name], cv=5, scoring='neg_mean_squared_error')
                                grid_search.fit(X, y)
                                results.append({
                                    'Model': model_name,
                                    'Scale Type': scale_type,
                                    'Best Params': grid_search.best_params_,
                                    'Best Score': grid_search.best_score_
                                })

                        st.session_state.results = results

                        # 최적 모델 정보 저장
                        best_model_info = min(results, key=lambda x: x['Best Score'])
                        st.session_state.best_model_info = best_model_info

                        # Plotting the results
                        results_df = pd.DataFrame(results)
                        sorted_results_df = results_df.sort_values(by='Best Score', ascending=True)
                        fig, ax = plt.subplots(figsize=(12, 8))
                        ax.barh(sorted_results_df['Model'] + ' (' + sorted_results_df['Scale Type'] + ')', -sorted_results_df['Best Score'])
                        ax.set_xlabel('Best Score (Negative MSE)')
                        ax.set_ylabel('Model (Scale Type)')
                        ax.set_title('Model Performance Comparison')
                        ax.invert_yaxis()  # To have the best score on top
                        st.pyplot(fig)

                        # Plot 이미지를 세션에 저장
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png")
                        buf.seek(0)
                        st.session_state.model_performance_plot = buf

                    except Exception as e:
                        st.error(f"모델을 선택하고 튜닝하는 중 오류가 발생했습니다: {e}")

 
            # Step 4: Model Performance Visualization
            if step_key == "Step 4":
                st.markdown("---")
                st.markdown("### 모델링 진행 버튼을 누르면 모델링이 시작됩니다")
                if st.button("모델링 진행"):
                    try:
                        with st.spinner("AUTOGEN이 답변 생성 중"):
                            time.sleep(3)
                        st.write("### 최종 모델 성능 시각화")
 
                        # 최적 모델 및 데이터셋 선택
                        best_model_info = st.session_state.best_model_info
                        best_model_name = best_model_info['Model']
                        scale_type = best_model_info['Scale Type']
                        best_params = best_model_info['Best Params']
 
                        st.write(f"최적 모델: {best_model_name} ({scale_type})")
                        st.write(f"최적 파라미터: {best_params}")
 
                        scaled_df = st.session_state.min_max_scaled if scale_type == 'MinMax Scaled' else st.session_state.standard_scaled
                        X = scaled_df.drop(columns=['EndPoint'])
                        y = scaled_df['EndPoint']
 
                        # 최적 모델 훈련
                        if best_model_name == 'Decision Tree':
                            best_model = DecisionTreeRegressor(**best_params)
                        elif best_model_name == 'Random Forest':
                            best_model = RandomForestRegressor(**best_params)
                        elif best_model_name == 'Gradient Boosting':
                            best_model = GradientBoostingRegressor(**best_params)
                        elif best_model_name == 'Support Vector Machine':
                            best_model = SVR(**best_params)
 
                        best_model.fit(X, y)
                        predictions = best_model.predict(X)
 
                        # 성능 시각화 (예측값 vs 실제값)
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.scatter(y, predictions, alpha=0.5)
                        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
                        ax.set_xlabel('Actual')
                        ax.set_ylabel('Predicted')
                        ax.set_title('Actual vs Predicted')
                        st.pyplot(fig)

                        # 최종 성능 시각화 이미지를 세션에 저장
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png")
                        buf.seek(0)
                        st.session_state.final_performance_plot = buf
 
                    except Exception as e:
                        st.error(f"최종 모델 성능을 시각화하는 중 오류가 발생했습니다: {e}")


            


def report_generation_page():
    """Report Generation page content."""
    st.title("Report Generation")
    st.markdown("---")
    st.markdown("### 1. Modeling Purpose")
    st.write("HN(Heavy Naptha)의 End Point를 다른 공정변수로 예측하고 싶어")

    st.markdown("---")
    st.markdown("### 2. Modeling 방향성")
    st.write("공정데이터 파일에서 HN의 End Point열을 제외한 나머지 열들로 HN의 End Point를 예측 ")
    st.write("")
    st.markdown("---")
    st.markdown("### 3. Modeling 결과")
    st.write("### 3-1 데이터 시각화")
    st.write("### DataFrame Describe")
    st.write(st.session_state.df_describe)
    st.write(f"### Box Plot")
    st.image(st.session_state['box_plot_0'].getvalue())
    st.write("### 3-2 데이터 전처리")
    st.write(f"### 데이터의 결측치 개수: {st.session_state.missing_values}")
    st.write(f"### 데이터의  이상치 개수: {st.session_state.num_outliers}")
    st.write("### 3-3 모델 선택 및 튜닝")
    st.image(st.session_state.model_performance_plot.getvalue())
    st.write("### 3-4 최종 모델 성능")
    st.image(st.session_state.final_performance_plot.getvalue())
    st.markdown("---")
    st.markdown("### 4. AUTOGEN 대화요약")
    st.markdown("---")
    if st.session_state.chat_histories:
        st.subheader("Modeling Chat History")
        
        col1, col2 = st.columns(2)
        with col1:
            if "Step 1" in st.session_state.chat_histories:
                st.write("**Step 1: Visualization**")
                for message in st.session_state.chat_histories["Step 1"]:
                    if message["role"] == "user":
                        st.write(f"You: {message['content']}")
                    else:
                        st.write(f"Assistant: {message['content']}")

        with col2:
            if "Step 2" in st.session_state.chat_histories:
                st.write("**Step 2: Preprocessing**")
                for message in st.session_state.chat_histories["Step 2"]:
                    if message["role"] == "user":
                        st.write(f"You: {message['content']}")
                    else:
                        st.write(f"Assistant: {message['content']}")

        col3, col4 = st.columns(2)
        with col3:
            if "Step 3" in st.session_state.chat_histories:
                st.write("**Step 3: Model Selection**")
                for message in st.session_state.chat_histories["Step 3"]:
                    if message["role"] == "user":
                        st.write(f"You: {message['content']}")
                    else:
                        st.write(f"Assistant: {message['content']}")

        with col4:
            if "Step 4" in st.session_state.chat_histories:
                st.write("**Step 4: Validation**")
                for message in st.session_state.chat_histories["Step 4"]:
                    if message["role"] == "user":
                        st.write(f"You: {message['content']}")
                    else:
                        st.write(f"Assistant: {message['content']}")
    else:
        st.write("모델링을 완료하고 눌러주세요")

def sidebar_menu():
    """Sidebar menu for main menu and sub-menu."""
    with st.sidebar:
        selected = option_menu("Main Menu", ["Main_Page", 'Detail_Modeling', 'Report_Generation'], 
                               icons=['house', 'gear'], menu_icon="cast", default_index=0)

        sub_selected = None
        if selected == 'Detail_Modeling' and st.session_state.confirmed:
            options = []
            for step, label in zip(['step1', 'step2', 'step3', 'step4'], 
                                   ["Step 1: Visualization", "Step 2: Preprocessing", "Step 3: Model Selection", "Step 4: Validation"]):
                if st.session_state[f"{step}_done"]:
                    options.append(f"{label} ✅")
                else:
                    options.append(label)
                
            sub_selected = option_menu("Modeling Steps", options, icons=['columns', 'bar-chart', 'check-square', 'check-square'], 
                                       menu_icon="list", default_index=0)
        return selected, sub_selected

def main():
    """Main function to run the app."""
    initialize_session_state()
    selected, sub_selected = sidebar_menu()
    
    if selected == "Main_Page":
        main_page()
    elif selected == "Detail_Modeling" and st.session_state.confirmed:
        detail_modeling_page(sub_selected)
    elif selected == "Report_Generation":
        report_generation_page()

if __name__ == "__main__":
    main()
