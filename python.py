import streamlit as st
import json
import math
import io
import time
from typing import List, Dict, Any, Optional

# --- Cấu Hình Gemini API (Gemini API Configuration) ---
# Sử dụng mô hình gemini-2.5-flash-preview-05-20 cho việc trích xuất (JSON Schema) và phân tích
API_KEY = "" # API Key sẽ được Canvas cung cấp tự động.
API_URL_BASE = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"

# --- Hàm Helpers cho API (API Helper Functions) ---

def fetch_with_retry(payload: Dict[str, Any], max_retries: int = 5) -> Optional[Dict[str, Any]]:
    """Thực hiện gọi API với cơ chế exponential backoff."""
    headers = {'Content-Type': 'application/json'}
    
    for attempt in range(max_retries):
        try:
            # Tên biến __api_key được sử dụng thay cho API_KEY để tích hợp với Canvas
            response = st.runtime.legacy_caching.request(
                API_URL_BASE,
                method='POST',
                headers=headers,
                data=json.dumps(payload),
                params={'key': API_KEY}
            )

            response.raise_for_status()
            return response.json()
        except Exception as e:
            if attempt < max_retries - 1:
                sleep_time = 2 ** attempt
                #st.warning(f"Lỗi khi gọi API (lần {attempt + 1}/{max_retries}). Thử lại sau {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                st.error(f"Lỗi: Không thể gọi API sau {max_retries} lần thử. Chi tiết: {e}")
                return None
    return None

def ai_extract_data(document_text: str) -> Optional[Dict[str, float]]:
    """Sử dụng Gemini để trích xuất dữ liệu tài chính dưới dạng JSON có cấu trúc."""
    
    system_prompt = (
        "Bạn là một chuyên gia phân tích tài chính. Hãy trích xuất các thông tin tài chính sau từ văn bản kế hoạch kinh doanh "
        "đã cung cấp và trả về dưới dạng đối tượng JSON. Đảm bảo tất cả các giá trị đều là số (number) "
        "hoặc số thập phân (decimal). Các giá trị WACC và TaxRate phải ở dạng thập phân (ví dụ: 0.10 cho 10%)."
    )

    user_query = f"Văn bản kế hoạch kinh doanh:\n\n---\n{document_text}\n---"

    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "InitialInvestment": {"type": "NUMBER", "description": "Tổng vốn đầu tư ban đầu của dự án (số âm)."},
                    "ProjectLifeYears": {"type": "INTEGER", "description": "Dòng đời dự án theo năm."},
                    "AnnualRevenue": {"type": "NUMBER", "description": "Doanh thu hàng năm (giả định là cố định)."},
                    "AnnualCost": {"type": "NUMBER", "description": "Chi phí hoạt động hàng năm (chưa bao gồm thuế)."},
                    "WACC": {"type": "NUMBER", "description": "Chi phí vốn bình quân (WACC) dưới dạng thập phân (ví dụ: 0.12)."},
                    "TaxRate": {"type": "NUMBER", "description": "Thuế suất doanh nghiệp dưới dạng thập phân (ví dụ: 0.20)."},
                },
                "required": ["InitialInvestment", "ProjectLifeYears", "AnnualRevenue", "AnnualCost", "WACC", "TaxRate"]
            }
        }
    }

    result = fetch_with_retry(payload)
    if result and result.get('candidates'):
        try:
            json_text = result['candidates'][0]['content']['parts'][0]['text']
            return json.loads(json_text)
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            st.error(f"Lỗi phân tích cú pháp JSON từ AI. Vui lòng kiểm tra nội dung file Word của bạn. Chi tiết lỗi: {e}")
            return None
    return None

def ai_analyze_results(npv: float, irr: float, pp: float, dpp: float, cash_flow_table: List[Dict[str, Any]]) -> str:
    """Sử dụng Gemini để phân tích các chỉ số hiệu quả dự án."""
    
    system_prompt = (
        "Bạn là một Giám đốc Tài chính (CFO) cấp cao. Dựa trên các chỉ số tài chính và bảng dòng tiền được cung cấp, "
        "hãy đưa ra một bản phân tích chuyên nghiệp và toàn diện về hiệu quả kinh tế của dự án. "
        "Đánh giá tính khả thi (thông qua NPV và IRR), rủi ro (thông qua thời gian hoàn vốn), và đưa ra khuyến nghị rõ ràng. "
        "Viết toàn bộ nội dung phân tích bằng tiếng Việt."
    )

    analysis_data = {
        "NPV": f"{npv:,.2f}",
        "IRR": f"{irr*100:,.2f}%",
        "PaybackPeriod_Years": f"{pp:,.2f}",
        "DiscountedPaybackPeriod_Years": f"{dpp:,.2f}",
        "CashFlowTableSnapshot": cash_flow_table
    }

    user_query = f"Phân tích dự án với dữ liệu sau:\n\n{json.dumps(analysis_data, indent=2)}"

    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
    }
    
    result = fetch_with_retry(payload)
    if result and result.get('candidates'):
        try:
            return result['candidates'][0]['content']['parts'][0]['text']
        except (KeyError, IndexError) as e:
            return "Không thể lấy kết quả phân tích từ AI. Vui lòng thử lại."
    return "Lỗi kết nối hoặc không có phản hồi từ AI."

# --- Hàm Tính Toán Tài Chính (Financial Calculation Functions) ---

def calculate_npv(rate: float, cash_flows: List[float]) -> float:
    """Tính Giá trị Hiện tại Thuần (Net Present Value - NPV)."""
    npv = 0.0
    for t, cf in enumerate(cash_flows):
        # t = 0 là năm đầu tư, t > 0 là dòng tiền từ hoạt động
        npv += cf / (1 + rate) ** t
    return npv

def calculate_irr(cash_flows: List[float]) -> float:
    """
    Tính Tỷ suất Hoàn vốn Nội bộ (Internal Rate of Return - IRR).
    Lưu ý: Không thể sử dụng NumPy.IRR, nên sẽ sử dụng phương pháp đoán/lặp đơn giản
    hoặc dùng giá trị ước tính an toàn (giả định IRR = WACC + 2% nếu NPV > 0).
    Trong môi trường thực tế, cần thư viện như numpy.
    Ở đây, tôi sẽ dùng phương pháp xấp xỉ đơn giản:
    """
    if not cash_flows or cash_flows[0] >= 0:
        return 0.0 # Không có đầu tư ban đầu hoặc dòng tiền không hợp lệ

    # Phương pháp xấp xỉ đơn giản: Thử các mức lãi suất từ 0% đến 100%
    low_rate, high_rate = 0.0, 1.0
    
    # Kiểm tra giới hạn
    if calculate_npv(low_rate, cash_flows) < 0:
        return -1.0 # Dự án NPV âm ngay cả ở 0%

    for _ in range(100): # 100 lần lặp để tìm kiếm nhị phân
        mid_rate = (low_rate + high_rate) / 2
        npv = calculate_npv(mid_rate, cash_flows)
        if abs(npv) < 0.0001:
            return mid_rate
        if npv > 0:
            low_rate = mid_rate
        else:
            high_rate = mid_rate
    return low_rate

def calculate_payback_periods(initial_investment: float, cash_flows_operating: List[float], wacc: float) -> tuple[float, float]:
    """Tính Thời gian Hoàn vốn (PP) và Thời gian Hoàn vốn có chiết khấu (DPP)."""
    
    # Dòng tiền hoàn vốn (chỉ tính dòng tiền dương sau năm 0)
    cumulative_cf = -initial_investment
    payback_period = 0.0
    
    # Dòng tiền chiết khấu hoàn vốn
    cumulative_dcf = -initial_investment
    discounted_payback_period = 0.0
    
    cf_data = []

    for t, cf in enumerate(cash_flows_operating, start=1):
        
        # 1. Tính PP (Payback Period)
        if payback_period == 0.0:
            if cumulative_cf + cf >= 0:
                # Tính phần năm
                payback_period = t - 1 + abs(cumulative_cf) / cf
            else:
                cumulative_cf += cf

        # 2. Tính DPP (Discounted Payback Period)
        discount_factor = 1 / (1 + wacc) ** t
        dcf = cf * discount_factor

        if discounted_payback_period == 0.0:
            if cumulative_dcf + dcf >= 0:
                # Tính phần năm
                discounted_payback_period = t - 1 + abs(cumulative_dcf) / dcf
            else:
                cumulative_dcf += dcf
                
    return payback_period, discounted_payback_period


# --- Giao Diện Streamlit (Streamlit UI) ---

def main():
    st.set_page_config(
        page_title="Đánh Giá Phương Án Kinh Doanh (Streamlit & Gemini)",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("💰 Ứng Dụng Đánh Giá Hiệu Quả Dự Án Kinh Doanh")
    st.markdown("""
    Ứng dụng sử dụng mô hình Gemini để **trích xuất thông tin tài chính** từ văn bản và **phân tích các chỉ số** hiệu quả dự án.
    """)
    st.divider()

    # --- Phần 1: Tải File và Lọc Dữ Liệu AI ---
    
    st.header("1. Lọc Dữ Liệu Tài Chính từ Văn Bản")

    uploaded_file = st.file_uploader(
        "Tải lên File Kế Hoạch Kinh Doanh (Khuyến nghị sử dụng file .txt hoặc .md, hoặc paste nội dung từ Word):",
        type=['txt', 'md', 'docx']
    )

    document_text = ""
    if uploaded_file is not None:
        try:
            # Đọc nội dung file. Lưu ý: Docx reader không hoạt động trong môi trường này.
            # Ta sẽ đọc file dưới dạng text thuần túy.
            document_text = uploaded_file.getvalue().decode("utf-8")
        except Exception:
            # Nếu không phải UTF-8, thử đọc nhị phân và hiển thị cảnh báo
            st.warning("Không thể giải mã file dưới dạng UTF-8. Vui lòng đảm bảo nội dung là văn bản thuần túy.")
            document_text = "Nội dung không đọc được."
            
        st.subheader("Nội dung Văn bản được tải lên:")
        st.text_area("Văn bản gốc", document_text, height=300)

    # Nút bấm để thực hiện trích xuất dữ liệu
    if st.button("Trích Xuất Dữ Liệu Tài Chính Bằng AI", key="extract_button"):
        if document_text and document_text != "Nội dung không đọc được.":
            with st.spinner("AI đang đọc và trích xuất các thông số tài chính..."):
                extracted_data = ai_extract_data(document_text)
            
            if extracted_data:
                # Lưu dữ liệu vào session state để dùng cho các bước sau
                st.session_state['extracted_data'] = extracted_data
                st.success("Trích xuất dữ liệu thành công!")
            else:
                st.error("Không thể trích xuất dữ liệu. Vui lòng kiểm tra lại nội dung file và thử lại.")
        else:
            st.warning("Vui lòng tải lên một file hoặc đảm bảo nội dung văn bản không trống.")
            
    # --- Hiển thị và Chỉnh Sửa Dữ Liệu Đã Lọc ---
    
    if 'extracted_data' in st.session_state:
        st.subheader("Tham Số Dự Án (Đã Lọc và Có Thể Chỉnh Sửa):")
        
        data = st.session_state['extracted_data']
        
        # Tạo giao diện nhập liệu để người dùng kiểm tra và chỉnh sửa
        col1, col2, col3 = st.columns(3)
        
        with col1:
            data['InitialInvestment'] = st.number_input(
                "Vốn Đầu Tư Ban Đầu (C0, Giá trị âm):", 
                value=float(data.get('InitialInvestment', 0)), 
                step=100000.0, 
                format="%.2f",
                key='C0_input'
            )
            data['AnnualRevenue'] = st.number_input(
                "Doanh Thu Hàng Năm (Gross Revenue):", 
                value=float(data.get('AnnualRevenue', 0)), 
                step=100000.0, 
                format="%.2f",
                key='Rev_input'
            )
        
        with col2:
            data['ProjectLifeYears'] = st.number_input(
                "Dòng Đời Dự Án (Năm):", 
                value=int(data.get('ProjectLifeYears', 0)), 
                min_value=1, 
                step=1,
                key='Life_input'
            )
            data['AnnualCost'] = st.number_input(
                "Chi Phí Hàng Năm (Operating Cost):", 
                value=float(data.get('AnnualCost', 0)), 
                step=100000.0, 
                format="%.2f",
                key='Cost_input'
            )
            
        with col3:
            data['WACC'] = st.number_input(
                "WACC (Chi Phí Vốn, dạng thập phân):", 
                value=float(data.get('WACC', 0.10)), 
                min_value=0.001, 
                max_value=1.0, 
                step=0.01,
                format="%.4f",
                key='WACC_input'
            )
            data['TaxRate'] = st.number_input(
                "Thuế Suất (dạng thập phân):", 
                value=float(data.get('TaxRate', 0.20)), 
                min_value=0.0, 
                max_value=1.0, 
                step=0.01,
                format="%.4f",
                key='Tax_input'
            )

        # Cập nhật lại session state sau khi chỉnh sửa
        st.session_state['extracted_data'] = data
        
        # --- Phần 2 & 3: Xây Dựng Bảng Dòng Tiền & Tính Toán Chỉ Số ---
        st.divider()
        st.header("2 & 3. Xây Dựng Dòng Tiền và Tính Toán Chỉ Số Hiệu Quả")

        # Chuẩn bị dữ liệu
        C0 = data['InitialInvestment']
        N = int(data['ProjectLifeYears'])
        R = data['AnnualRevenue']
        OC = data['AnnualCost']
        WACC = data['WACC']
        T = data['TaxRate']
        
        # Dòng tiền hàng năm (Operating Cash Flow - OCF)
        EBIT = R - OC # Lợi nhuận trước thuế và lãi suất (Giả định không có khấu hao)
        Tax = EBIT * T if EBIT > 0 else 0
        NPAT = EBIT - Tax # Lợi nhuận sau thuế (Net Profit After Tax)
        # OCF = NPAT + Khấu hao - Thay đổi vốn lưu động (Giả định Khấu hao và Thay đổi vốn lưu động = 0)
        OCF = NPAT 

        # Xây dựng bảng dòng tiền
        cash_flow_table: List[Dict[str, Any]] = []
        cash_flows_for_npv_irr = [] # Dòng tiền ròng cho tính toán NPV/IRR
        
        # Năm 0 (Đầu tư ban đầu)
        cash_flow_table.append({
            'Year': 0,
            'Revenue': 0.0,
            'Operating Cost': 0.0,
            'EBIT': 0.0,
            'Tax': 0.0,
            'OCF': 0.0,
            'Investment': C0,
            'Net Cash Flow': C0,
            'Discount Factor': 1.0,
            'Discounted Cash Flow': C0
        })
        cash_flows_for_npv_irr.append(C0)

        # Các năm hoạt động (Năm 1 đến N)
        for year in range(1, N + 1):
            discount_factor = 1 / (1 + WACC) ** year
            net_cf = OCF # Net Cash Flow = OCF (Giả định không có vốn lưu động cuối kỳ)
            
            # Giả định: Thu hồi vốn đầu tư (Terminal Value) bằng 0, hoặc có giá trị thanh lý. 
            # Để đơn giản, giả định không có giá trị thanh lý.
            
            cash_flow_table.append({
                'Year': year,
                'Revenue': R,
                'Operating Cost': OC,
                'EBIT': EBIT,
                'Tax': Tax,
                'OCF': OCF,
                'Investment': 0.0,
                'Net Cash Flow': net_cf,
                'Discount Factor': discount_factor,
                'Discounted Cash Flow': net_cf * discount_factor
            })
            cash_flows_for_npv_irr.append(net_cf)
            
        # Hiển thị bảng dòng tiền
        st.subheader("Bảng Dòng Tiền Dự Án (Cash Flow Statement)")
        st.dataframe(
            cash_flow_table,
            column_order=['Year', 'Revenue', 'Operating Cost', 'EBIT', 'Tax', 'OCF', 'Investment', 'Net Cash Flow', 'Discount Factor', 'Discounted Cash Flow'],
            hide_index=True,
            use_container_width=True,
            column_config={
                'Year': st.column_config.NumberColumn("Năm", format="%d"),
                'Revenue': st.column_config.NumberColumn("Doanh Thu", format="%.0f"),
                'Operating Cost': st.column_config.NumberColumn("Chi Phí HĐ", format="%.0f"),
                'EBIT': st.column_config.NumberColumn("EBIT", format="%.0f"),
                'Tax': st.column_config.NumberColumn("Thuế", format="%.0f"),
                'OCF': st.column_config.NumberColumn("OCF", format="%.0f"),
                'Investment': st.column_config.NumberColumn("Vốn Đầu Tư", format="%.0f"),
                'Net Cash Flow': st.column_config.NumberColumn("Dòng Tiền Ròng (CF)", format="%.0f"),
                'Discount Factor': st.column_config.NumberColumn("Hệ Số Chiết Khấu", format="%.4f"),
                'Discounted Cash Flow': st.column_config.NumberColumn("Dòng Tiền Chiết Khấu (DCF)", format="%.0f"),
            }
        )

        # Tính toán các chỉ số
        npv = calculate_npv(WACC, cash_flows_for_npv_irr)
        irr = calculate_irr(cash_flows_for_npv_irr)
        
        # Dòng tiền hoạt động (bỏ C0)
        cash_flows_operating = cash_flows_for_npv_irr[1:] 
        pp, dpp = calculate_payback_periods(abs(C0), cash_flows_operating, WACC)
        
        st.subheader("Chỉ Số Đánh Giá Hiệu Quả Dự Án")
        
        kpi_cols = st.columns(4)
        
        kpi_cols[0].metric("NPV (Giá trị hiện tại thuần)", f"{npv:,.0f}")
        kpi_cols[1].metric("IRR (Tỷ suất hoàn vốn nội bộ)", f"{irr*100:,.2f}%", delta=f"WACC: {WACC*100:,.2f}%")
        kpi_cols[2].metric("PP (Thời gian hoàn vốn)", f"{pp:,.2f} năm", delta_color="off")
        kpi_cols[3].metric("DPP (Hoàn vốn có chiết khấu)", f"{dpp:,.2f} năm", delta_color="off")
        
        # Lưu kết quả vào session state
        st.session_state['kpis'] = {
            'npv': npv, 
            'irr': irr, 
            'pp': pp, 
            'dpp': dpp, 
            'cash_flow_table_snapshot': cash_flow_table
        }
        
        # --- Phần 4: Phân Tích Chỉ Số Bằng AI ---
        st.divider()
        st.header("4. Phân Tích Chuyên Sâu Của AI")
        
        if st.button("Yêu Cầu AI Phân Tích Hiệu Quả Dự Án", key="analyze_button"):
            if 'kpis' in st.session_state:
                with st.spinner("AI đang phân tích các chỉ số tài chính và lập báo cáo..."):
                    kpis = st.session_state['kpis']
                    analysis_text = ai_analyze_results(
                        kpis['npv'], 
                        kpis['irr'], 
                        kpis['pp'], 
                        kpis['dpp'], 
                        kpis['cash_flow_table_snapshot']
                    )
                
                st.subheader("Báo Cáo Phân Tích Của CFO AI")
                st.markdown(analysis_text)
            else:
                st.warning("Vui lòng thực hiện Bước 1 và 2-3 trước khi yêu cầu phân tích.")

    st.divider()
    st.markdown("---")
    st.info("Lưu ý: Các chỉ số IRR được tính toán bằng phương pháp xấp xỉ đơn giản. Trong môi trường thực tế, hãy sử dụng thư viện tài chính chuyên nghiệp.")

if __name__ == "__main__":
    main()
