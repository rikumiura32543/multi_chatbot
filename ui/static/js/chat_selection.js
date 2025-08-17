/**
 * チャットボット選択画面 - JavaScript機能
 * マルチチャットボット管理UI
 */

// グローバル変数
let chatbots = [];
let deleteTargetId = null;

// ページ読み込み時の初期化
document.addEventListener('DOMContentLoaded', function() {
    // モーダルの初期状態を確実に非表示にする
    resetModals();
    loadChatbots();
    setupEventListeners();
});

/**
 * モーダルの初期化（非表示状態にリセット）
 */
function resetModals() {
    const createModal = document.getElementById('createModal');
    const deleteModal = document.getElementById('deleteModal');
    
    if (createModal) {
        createModal.style.display = 'none';
    }
    if (deleteModal) {
        deleteModal.style.display = 'none';
    }
    
    // 削除対象IDもリセット
    deleteTargetId = null;
}

/**
 * イベントリスナーの設定
 */
function setupEventListeners() {
    // モーダル外クリックで閉じる
    window.onclick = function(event) {
        const createModal = document.getElementById('createModal');
        const deleteModal = document.getElementById('deleteModal');
        
        if (event.target === createModal) {
            closeCreateModal();
        }
        if (event.target === deleteModal) {
            closeDeleteModal();
        }
    };
    
    // ESCキーでモーダルを閉じる
    document.addEventListener('keydown', function(event) {
        if (event.key === 'Escape') {
            closeCreateModal();
            closeDeleteModal();
        }
    });
}

/**
 * チャットボット一覧の読み込み
 */
async function loadChatbots() {
    console.log('loadChatbots開始');
    try {
        showLoading();
        console.log('Loading表示完了');
        
        const response = await fetch('/api/chatbots/list');
        console.log('API response:', response.status, response.statusText);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        chatbots = await response.json();
        console.log('Chatbots loaded:', chatbots.length, 'items');
        renderChatbots();
        console.log('renderChatbots完了');
        updateCreateButton();
        console.log('updateCreateButton完了');
        
    } catch (error) {
        console.error('チャットボット一覧読み込みエラー:', error);
        showError('チャットボット一覧の読み込みに失敗しました。');
    }
    console.log('loadChatbots完了');
}

/**
 * チャットボット一覧の表示
 */
function renderChatbots() {
    const grid = document.getElementById('chatbotGrid');
    
    if (chatbots.length === 0) {
        grid.innerHTML = `
            <div class="loading-container">
                <p>まだチャットボットが作成されていません</p>
                <p>右上の「新規作成」ボタンから最初のチャットボットを作成してください</p>
            </div>
        `;
        return;
    }
    
    grid.innerHTML = chatbots.map(chatbot => `
        <div class="chatbot-card" data-id="${chatbot.id}">
            <div class="chatbot-header">
                <h3 class="chatbot-name">${escapeHtml(chatbot.name)}</h3>
                <span class="rag-status ${chatbot.rag_configured ? 'text-primary' : 'text-muted'}">
                    ${chatbot.rag_configured ? 'RAG設定済み' : 'RAG未設定'}
                </span>
            </div>
            
            <div class="chatbot-info">
                <p><strong>説明:</strong> ${escapeHtml(chatbot.description || '説明なし')}</p>
                <p><strong>処理済み文書:</strong> ${chatbot.document_count}件</p>
                ${chatbot.folder_paths && chatbot.folder_paths.length > 0 ? 
                    `<p><strong>参照フォルダ:</strong> ${chatbot.folder_paths.length}個</p>` : ''
                }
            </div>
            
            <div class="chatbot-meta">
                <span>作成: ${formatDate(chatbot.created_at)}</span>
                <span>更新: ${formatDate(chatbot.updated_at)}</span>
            </div>
            
            <div class="chatbot-actions">
                <button class="btn btn-primary" onclick="openChat('${chatbot.id}')">
                    チャット開始
                </button>
                <button class="btn" onclick="openSettings('${chatbot.id}')">
                    RAG設定
                </button>
                <button class="btn btn-danger" onclick="showDeleteModal('${chatbot.id}', '${escapeHtml(chatbot.name)}')">
                    削除
                </button>
            </div>
        </div>
    `).join('');
}

/**
 * 新規作成ボタンの状態更新
 */
function updateCreateButton() {
    const createBtn = document.getElementById('createBtn');
    const maxChatbots = 5;
    
    if (chatbots.length >= maxChatbots) {
        createBtn.disabled = true;
        createBtn.innerHTML = `上限達成 (${chatbots.length}/${maxChatbots})`;
        createBtn.title = '最大5個まで作成可能です';
    } else {
        createBtn.disabled = false;
        createBtn.innerHTML = `新規作成 (${chatbots.length}/${maxChatbots})`;
        createBtn.title = '';
    }
}

/**
 * 新規チャットボット作成モーダル表示
 */
function createNewChatbot() {
    console.log('createNewChatbot called, current chatbots count:', chatbots.length);
    
    if (chatbots.length >= 5) {
        showError('チャットボットは最大5個まで作成可能です。');
        return;
    }
    
    // フォームリセット
    const form = document.getElementById('createForm');
    const modal = document.getElementById('createModal');
    const nameField = document.getElementById('chatbotName');
    
    console.log('Form element:', form);
    console.log('Modal element:', modal);
    console.log('Name field element:', nameField);
    
    if (form) form.reset();
    if (modal) {
        modal.style.display = 'flex';
        console.log('Modal display set to flex');
    }
    if (nameField) {
        nameField.focus();
        console.log('Name field focused');
    }
}

/**
 * 新規作成モーダルを閉じる
 */
function closeCreateModal() {
    console.log('closeCreateModal開始');
    const modal = document.getElementById('createModal');
    console.log('Modal element:', modal);
    
    if (modal) {
        modal.style.display = 'none';
        console.log('Modal display set to none');
    }
    
    // ボタンの状態をリセット
    const submitButton = document.querySelector('button[form="createForm"]');
    console.log('Submit button:', submitButton);
    if (submitButton) {
        submitButton.disabled = false;
        submitButton.innerHTML = '作成';
        submitButton.style.backgroundColor = '';
        console.log('Submit button reset');
    }
    console.log('closeCreateModal完了');
}

/**
 * 新規作成フォーム送信
 */
async function submitCreateForm(event) {
    console.log('submitCreateForm called', event);
    event.preventDefault();
    
    // 送信ボタンの特定（フォーム外にある場合の対応）
    let submitButton = event.target.querySelector('button[type="submit"]');
    if (!submitButton) {
        submitButton = document.querySelector('button[form="createForm"]');
    }
    console.log('Submit button found:', submitButton);
    
    const originalText = submitButton ? submitButton.innerHTML : 'Submit';
    
    const formData = new FormData(event.target);
    const data = {
        name: formData.get('name').trim(),
        description: formData.get('description').trim()
    };
    
    // バリデーション
    if (!data.name) {
        showError('チャットボット名を入力してください。');
        return;
    }
    
    if (chatbots.some(bot => bot.name === data.name)) {
        showError('同じ名前のチャットボットが既に存在します。');
        return;
    }
    
    try {
        // ボタンを無効化して処理中表示
        if (submitButton) {
            submitButton.disabled = true;
            submitButton.innerHTML = '作成中...';
        }
        
        const response = await fetch('/api/chatbots/create', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'チャットボット作成に失敗しました');
        }
        
        const newChatbot = await response.json();
        
        // 成功メッセージ表示とボタンの状態変更
        if (submitButton) {
            submitButton.innerHTML = '✅ 作成完了';
            submitButton.style.backgroundColor = '#4CAF50';
            console.log('作成ボタンを成功状態に変更');
        }
        showSuccess(`チャットボット "${newChatbot.name}" を作成しました。`);
        console.log('成功メッセージ表示完了');
        
        // モーダルを閉じる前に少し待つ（成功メッセージが見えるように）
        console.log('1.5秒後にモーダルを閉じるタイマーを設定');
        setTimeout(() => {
            console.log('===== タイマー実行開始 =====');
            console.log('About to close create modal and reload chatbots');
            try {
                closeCreateModal();
                console.log('closeCreateModal実行完了');
                loadChatbots(); // 一覧を再読み込み（awaitを削除）
                console.log('loadChatbots実行完了');
            } catch (error) {
                console.error('タイマー内でエラー発生:', error);
            }
            console.log('===== タイマー実行終了 =====');
        }, 1500);
        
    } catch (error) {
        console.error('チャットボット作成エラー:', error);
        showError(error.message);
    } finally {
        // ボタンを元に戻す（エラー時のみ）
        if (submitButton && submitButton.innerHTML !== '✅ 作成完了') {
            submitButton.disabled = false;
            submitButton.innerHTML = originalText;
            submitButton.style.backgroundColor = '';
        }
    }
}

/**
 * チャット画面を開く
 */
function openChat(chatbotId) {
    window.location.href = `/chat/${chatbotId}`;
}

/**
 * RAG設定画面を開く
 */
function openSettings(chatbotId) {
    window.location.href = `/settings/${chatbotId}`;
}

/**
 * 削除確認モーダル表示
 */
function showDeleteModal(chatbotId, chatbotName) {
    console.log('showDeleteModal called:', chatbotId, chatbotName);
    deleteTargetId = chatbotId;
    
    const targetNameElement = document.getElementById('deleteTargetName');
    const modal = document.getElementById('deleteModal');
    
    console.log('Target name element:', targetNameElement);
    console.log('Delete modal:', modal);
    
    if (targetNameElement) {
        targetNameElement.textContent = chatbotName;
    }
    if (modal) {
        modal.style.display = 'flex';
        console.log('Delete modal display set to flex');
    }
}

/**
 * 削除確認モーダルを閉じる
 */
function closeDeleteModal() {
    console.log('closeDeleteModal開始');
    const modal = document.getElementById('deleteModal');
    console.log('Delete modal element:', modal);
    
    if (modal) {
        modal.style.display = 'none';
        console.log('Delete modal display set to none');
    }
    deleteTargetId = null;
    console.log('deleteTargetId reset to null');
    
    // ボタンの状態をリセット
    const deleteButton = document.querySelector('#deleteModal button.btn-danger');
    console.log('Delete button:', deleteButton);
    if (deleteButton) {
        deleteButton.disabled = false;
        deleteButton.innerHTML = '削除実行';
        deleteButton.style.backgroundColor = '';
        console.log('Delete button reset');
    }
    console.log('closeDeleteModal完了');
}

/**
 * チャットボット削除実行
 */
async function confirmDelete() {
    if (!deleteTargetId) return;
    
    const deleteButton = document.querySelector('#deleteModal button.btn-danger');
    console.log('Delete button found:', deleteButton);
    
    if (!deleteButton) {
        console.error('Delete button not found');
        showError('削除ボタンが見つかりません');
        return;
    }
    
    const originalText = deleteButton.innerHTML;
    
    try {
        // ボタンを無効化して処理中表示
        deleteButton.disabled = true;
        deleteButton.innerHTML = '削除中...';
        
        const response = await fetch(`/api/chatbots/${deleteTargetId}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'チャットボット削除に失敗しました');
        }
        
        // 成功メッセージ表示とボタンの状態変更
        deleteButton.innerHTML = '✅ 削除完了';
        deleteButton.style.backgroundColor = '#4CAF50';
        console.log('削除ボタンを成功状態に変更');
        showSuccess('チャットボットを削除しました。');
        console.log('削除成功メッセージ表示完了');
        
        // モーダルを閉じる前に少し待つ（成功メッセージが見えるように）
        console.log('1.5秒後にモーダルを閉じるタイマーを設定');
        setTimeout(() => {
            console.log('===== 削除タイマー実行開始 =====');
            console.log('About to close delete modal and reload chatbots');
            try {
                closeDeleteModal();
                console.log('closeDeleteModal実行完了');
                loadChatbots(); // 一覧を再読み込み（awaitを削除）
                console.log('loadChatbots実行完了');
            } catch (error) {
                console.error('削除タイマー内でエラー発生:', error);
            }
            console.log('===== 削除タイマー実行終了 =====');
        }, 1500);
        
    } catch (error) {
        console.error('チャットボット削除エラー:', error);
        showError(error.message);
    } finally {
        // ボタンを元に戻す（エラー時のみ）
        if (deleteButton.innerHTML !== '✅ 削除完了') {
            deleteButton.disabled = false;
            deleteButton.innerHTML = originalText;
            deleteButton.style.backgroundColor = '';
        }
    }
}

/**
 * 読み込み表示
 */
function showLoading() {
    const grid = document.getElementById('chatbotGrid');
    grid.innerHTML = `
        <div class="loading-container">
            <div class="spinner"></div>
            <p>チャットボット一覧を読み込み中...</p>
        </div>
    `;
}

/**
 * 成功メッセージ表示
 */
function showSuccess(message) {
    showAlert(message, 'success');
}

/**
 * エラーメッセージ表示
 */
function showError(message) {
    showAlert(message, 'error');
}

/**
 * アラート表示
 */
function showAlert(message, type) {
    // 既存のアラートを削除
    const existingAlert = document.querySelector('.alert');
    if (existingAlert) {
        existingAlert.remove();
    }
    
    // 新しいアラートを作成
    const alert = document.createElement('div');
    alert.className = `alert alert-${type}`;
    alert.innerHTML = `
        ${escapeHtml(message)}
    `;
    
    // メインコンテンツの最初に挿入
    const mainContent = document.querySelector('.container') || document.querySelector('main') || document.body;
    console.log('Alert insertion target:', mainContent);
    
    if (mainContent && mainContent.firstChild) {
        mainContent.insertBefore(alert, mainContent.firstChild);
    } else if (mainContent) {
        mainContent.appendChild(alert);
    } else {
        document.body.appendChild(alert);
    }
    
    // 3秒後に自動削除
    setTimeout(() => {
        if (alert.parentNode) {
            alert.remove();
        }
    }, 3000);
}

/**
 * HTMLエスケープ
 */
function escapeHtml(text) {
    if (!text) return '';
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, function(m) { return map[m]; });
}

/**
 * 日時フォーマット
 */
function formatDate(dateString) {
    if (!dateString) return '不明';
    
    const date = new Date(dateString);
    const now = new Date();
    const diffTime = Math.abs(now - date);
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
    
    if (diffDays === 1) {
        return '今日';
    } else if (diffDays <= 7) {
        return `${diffDays}日前`;
    } else {
        return date.toLocaleDateString('ja-JP', {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        });
    }
}