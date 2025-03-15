<h1 align="center"><strong><span style="color: red;">Tuần 3: Text Normalization and Vector Semantics, Embeddings</span></strong><h1>

## Mục lục

### 1. Chuẩn hóa văn bản (Text Normalization)
- Chuẩn hóa văn bản là một chuỗi việc chuyển văn bản sang **dạng chuẩn**, **thuận tiện** để sử dụng trong các bài toán khác nhau:
  - **Bài toán sinh từ** (Text Generation) : giữ nhiều token nhất có thể, đưa vào các văn bản về chung một format. Ví dụ lùi đầu dòng, viết hoa đầu câu
  - **Bài toán phân loại cảm xúc** (Sentiment Classification) : loại bỏ những stop-words như the, a, to,... Giữ lại biểu tượng cảm xúc như :), :D
- Sentence segmentation (Tách câu) : chia văn bản thành các câu. Việc chia này có thể dựa vào dấu ".", "?", "!". Vấn đề khó khăn xảy ra như từ viết tắt trong Tiếng Anh sử dụng dấu chấm, ví dụ như Mr. hay Mrs.
- Tokenization (Tách token) : Chia văn bản thành các token 
- Lematization (Đưa về dạng từ gốc) : là việc xác định từ gốc của các từ
  - Ví dụ: say, said, saying -> **say**
  - Ưu điểm:  
    - Tìm kiếm tốt hơn: khi người dùng tìm kiếm văn bản từ **sing**, thuật toán có thể cùng tìm thêm từ **sang**, **sung**
    - Phân loại tốt hơn: chuẩn hóa về từ gốc giúp thu hẹp không gian phân tích và tạo ra độ chính xác cao hơn
  - Nhược điểm:
    - Đánh mất thông tin ngữ pháp: nếu bộ dữ liệu có sự mập mờ lớn thì việc xử lý này sẽ đánh mất thông tin ngữ pháp làm giảm độ chính xác của mô hình
  - Một số thư viện hay dùng
    - Natural Language Toolkit: https://www.nltk.org/
    - spaCy: https://spacy.io/
    - TextBlob: https://textblob.readthedocs.io/en/dev/
    - Stanford CoreNLP: https://stanfordnlp.github.io/CoreNLP/
- Stemming : Cắt hậu tố khỏi từ. Ít được sử dụng hơn Lemmzatiazation
- Lọc stop words : Lọc những từ hay xuất hiện và ít ngữ nghĩa như "the", "is", "at", "on", "which",...
- Word Correction (Sửa sai từ) : sai thứ tự chữ trong từ Tiếng Anh hoặc sai dấu trong Tiếng Việt
  - Ví dụ: happpy -> happy, azmaing -> amazing, intelliengt -> intelligent
  - Tìm từ trong từ điển từ có **khoảng cách** gần nhất. Khoảng cách này có thể sử dụng Edit Distance hoặc Jaccard Distance