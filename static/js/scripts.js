$(document).ready(function(){
    // アニメーション初期化
    $('.form-section, .results-container').addClass('animate-fade-in');
 
    // Socket.IOの接続を設定
    var socket = io.connect(window.location.protocol + '//' + document.domain + ':' + location.port);
 
    socket.on('connect', function() {
        console.log('Socket.IO Connected');
    });
 
    socket.on('error', function(error) {
        console.log('Socket.IO Error:', error);
    });
 
    // デバイスの判定
    function isMobile() {
        return window.innerWidth <= 768;
    }
 
    // カレンダーの初期化と位置設定
    function initializeDatepicker(element) {
        let $input = $(element);
        let position = $input.offset();
        let inputHeight = $input.outerHeight();
 
        $(element).datepicker({
            format: 'yyyy-mm-dd',
            autoclose: true,
            container: isMobile() ? 'body' : $input.parent(),
            orientation: isMobile() ? 'auto' : 'auto right',
            language: 'ja',
            todayHighlight: true,
            templates: {
                leftArrow: '<i class="fas fa-chevron-left"></i>',
                rightArrow: '<i class="fas fa-chevron-right"></i>'
            }
        }).on('show', function(e) {
            if (!isMobile()) {
                let $picker = $(this).datepicker('widget');
                $picker.css({
                    top: position.top + inputHeight + 5,
                    left: position.left
                });
            }
        });
    }
 
    // 各日付入力フィールドにカレンダーを初期化
    $('.datepicker').each(function() {
        initializeDatepicker(this);
    });
 
    // ウィンドウリサイズ時の再初期化
    let resizeTimer;
    $(window).resize(function() {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(function() {
            $('.datepicker').datepicker('destroy');
            $('.datepicker').each(function() {
                initializeDatepicker(this);
            });
        }, 250);
    });
 
    // フォームの送信をAjax化
    $('#mainForm').on('submit', function(e) {
        e.preventDefault(); // デフォルトのフォーム送信を防止
        
        var formData = new FormData(this);
        var submitAction = $('button[type="submit"]:focus').val();
        formData.append('submit_action', submitAction);
 
        if (submitAction === 'analyze') {
            // 分析実行ボタンのスピナー表示
            const submitBtn = $('button[name="submit_action"][value="analyze"]');
            submitBtn.prop('disabled', true)
                    .find('.spinner-border')
                    .removeClass('d-none');
 
            $.ajax({
                url: '/index',
                type: 'POST',
                data: formData,
                processData: false,
                contentType: false,
                success: function(response) {
                    if (response.redirect) {
                        window.location.href = response.redirect;
                    }
                },
                error: function(xhr, status, error) {
                    showNotification('エラーが発生しました: ' + error, 'error');
                },
                complete: function() {
                    submitBtn.prop('disabled', false)
                            .find('.spinner-border')
                            .addClass('d-none');
                }
            });
        } else if (submitAction === 'save') {
            $.ajax({
                url: '/index',
                type: 'POST',
                data: formData,
                processData: false,
                contentType: false,
                success: function(response) {
                    showNotification('設定を保存しました', 'success');
                },
                error: function(xhr, status, error) {
                    showNotification('設定の保存に失敗しました: ' + error, 'error');
                }
            });
        }
    });
 
    // 設定のロード処理
    $('.load-setting').click(function(e) {
        e.preventDefault();
        const settingId = $(this).data('id');
        const loadingBtn = $(this);
        
        // ロードボタンをスピナー表示に変更
        loadingBtn.html('<span class="spinner-border spinner-border-sm"></span> ロード中');
 
        $.ajax({
            url: `/load_setting/${settingId}`,
            type: 'GET',
            success: function(data) {
                // フォームフィールドのアップデート
                updateFormFields(data);
                // 隠しフィールドに設定IDを保存
                $('#loaded_setting_id').val(settingId);
                // 成功メッセージの表示
                showNotification('設定を正常にロードしました', 'success');
            },
            error: function() {
                showNotification('設定のロードに失敗しました', 'error');
            },
            complete: function() {
                // ロードボタンを元に戻す
                loadingBtn.html('<i class="fas fa-file mr-2"></i>ロード');
            }
        });
    });
 
    // フォームフィールドの更新
    function updateFormFields(data) {
        // 日付の設定
        $('#start_date').val(data.start_date);
        $('#end_date').val(data.end_date);
        
        // インデックスの設定
        $('input[name="indices"]').prop('checked', false);
        data.selected_indices.forEach(index => {
            $(`input[name="indices"][value="${index}"]`).prop('checked', true);
        });
        
        // コストの設定
        const spreads = data.spreads.map(Number);
        const swapsLong = data.swaps_long.map(Number);
        const swapsShort = data.swaps_short.map(Number);
        
        $('input[name="spread"]').each((index, element) => {
            if (index < spreads.length) {
                $(element).val(spreads[index]);
            }
        });
        
        $('input[name="swap_long"]').each((index, element) => {
            if (index < swapsLong.length) {
                $(element).val(swapsLong[index]);
            }
        });
        
        $('input[name="swap_short"]').each((index, element) => {
            if (index < swapsShort.length) {
                $(element).val(swapsShort[index]);
            }
        });
 
        // その他の設定
        $('#initial_margin').val(data.initial_margin);
        $('#min_lot').val(data.min_lot);
        $('#max_lot').val(data.max_lot);
        $('#handle_small_hedge').val(data.handle_small_hedge);
        $('#settlement_frequency').val(data.settlement_frequency);
        $('#optimize_weights').prop('checked', data.optimize_weights);
        $('#setting_name').val(data.name);
    }
 
    // 通知表示
    function showNotification(message, type) {
        const alertClass = type === 'success' ? 'alert-success' : 'alert-danger';
        const alert = $(`<div class="alert ${alertClass} alert-dismissible fade show" role="alert">
            ${message}
            <button type="button" class="close" data-dismiss="alert" aria-label="Close">
                <span aria-hidden="true">&times;</span>
            </button>
        </div>`);
        
        $('.container').prepend(alert);
        setTimeout(() => alert.alert('close'), 5000);
    }
 
    // 設定の削除処理
    $('#delete_setting_btn').click(function(e) {
        e.preventDefault();
        const settingId = $('#loaded_setting_id').val();
        
        if (!settingId) {
            showNotification('削除する設定を選択してください', 'error');
            return;
        }
        
        if (confirm('この設定を削除してもよろしいですか？')) {
            const btn = $(this);
            // 削除ボタンをスピナー表示に変更
            btn.prop('disabled', true).html('<span class="spinner-border spinner-border-sm"></span> 削除中');
 
            $.ajax({
                url: `/delete_setting/${settingId}`,
                type: 'POST',
                success: function(response) {
                    if (response.status === 'success') {
                        showNotification('設定を削除しました。', 'success');
                        // 設定を削除したらページをリロード
                        location.reload();
                    } else {
                        showNotification('設定の削除に失敗しました。', 'error');
                        btn.prop('disabled', false).html('<i class="fas fa-trash-alt mr-1"></i>削除');
                    }
                },
                error: function() {
                    showNotification('設定の削除に失敗しました。', 'error');
                    btn.prop('disabled', false).html('<i class="fas fa-trash-alt mr-1"></i>削除');
                }
            });
        }
    });
 
    // ログメッセージの受信と表示
    socket.on('log', function(data) {
        console.log('Received log:', data);
        $('#log-content').append(`<p><i class="fas fa-info-circle mr-2"></i>${data.message}</p>`);
        // 自動スクロール
        var logContainer = $('#log-content');
        logContainer.scrollTop(logContainer[0].scrollHeight);
    });
 
    // ツールチップの初期化
    $('[data-toggle="tooltip"]').tooltip();
 
    // スムーズスクロール
    $('a[href^="#"]').on('click', function(e) {
        e.preventDefault();
        const target = $(this).getAttribute('href');
        if (target.length) {
            $('html, body').animate({
                scrollTop: $(target).offset().top - 20
            }, 500);
        }
    });
 });