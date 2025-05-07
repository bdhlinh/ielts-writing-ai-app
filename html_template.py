css = '''
<style>
.chat-message {
    padding: 1.25rem;
    border-radius: 0.75rem;
    margin-bottom: 1rem;
    display: flex;
    min-height: 120px; /* ✅ đảm bảo mỗi khung có chiều cao tối thiểu */
    align-items: center;
}

.chat-message.user {
    background-color: #2b313e;
}

.chat-message.bot {
    background-color: #475063;
}

.chat-message.overall {
    background-color: #edadc2;
    font-size: 18px;
    font-weight: bold;
}

.chat-message .avatar {
    width: 20%;
}

.chat-message .avatar img {
    max-width: 78px;
    max-height: 78px;
    border-radius: 50%;
    object-fit: cover;
}

.chat-message .message {
    width: 100%;
    padding: 0 1.5rem;
    color: #fff;
    line-height: 1.6;           /* ✅ khoảng cách dòng hợp lý */
    white-space: pre-wrap;      /* ✅ xuống dòng đúng như nội dung */
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="message">{{MSG}}</div>
</div>
'''
overall_template = '''
<div class="chat-message overall">
    <div class="message">🏆 {{MSG}}</div>
</div>
'''
