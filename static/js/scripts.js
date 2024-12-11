$(document).ready(function(){
    // アニメーション初期化
    $('.form-section, .results-container').addClass('animate-fade-in');
 
    // Socket.IOの接続を設定
    var socket = io.connect(window.location.protocol + '//' + document.domain + ':' + location.port + '/logs');
 
    socket.on('connect', function() {
        console.log('SocketIO Connected');
    });
 
    socket.on('error', function(error) {
        console.log('SocketIO Error:', error);
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
        e.preventDefault();
 
        // フォームデータを作成する前にコンソールで値を確認
        const formValues = {
            initial_margin: $('#initial_margin').val(),
            min_lot: $('#min_lot').val(),
            max_lot: $('#max_lot').val(),
            handle_small_hedge: $('#handle_small_hedge').val(),
            settlement_frequency: $('#settlement_frequency').val()
        };
        console.log("Form Values before FormData:", formValues);
 
        var formData = new FormData(this);
        var submitAction = $('button[type="submit"]:focus').val();
        formData.append('submit_action', submitAction);
 
        // フォームデータの内容を確認
        console.log("Complete FormData entries:");
        for (var pair of formData.entries()) {
            console.log(pair[0] + ': ' + pair[1]);
        }
 
        // スプレッド、スワップ、レバレッジ、契約サイズの値を確認
        const assetNames = ['SP500', 'Nikkei', 'EuroStoxx', 'FTSE', 'DowJones', 'AUS200', 'HK50', 'SMI20', 'VIX'];
        console.log("Asset-specific values:");
        assetNames.forEach(asset => {
            console.log(`${asset}:`);
            console.log(`  spread_${asset}: ${formData.get('spread_' + asset)}`);
            console.log(`  swap_long_${asset}: ${formData.get('swap_long_' + asset)}`);
            console.log(`  swap_short_${asset}: ${formData.get('swap_short_' + asset)}`);
            console.log(`  leverage_${asset}: ${formData.get('leverage_' + asset)}`);
            console.log(`  contract_size_${asset}: ${formData.get('contract_size_' + asset)}`);
        });
 
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
                    console.log("Server Response:", response);
                    if (response.redirect) {
                        window.location.href = response.redirect;
                    }
                },
                error: function(xhr, status, error) {
                    console.error("Ajax Error:", error);
                    console.error("Status:", status);
                    console.error("Response:", xhr.responseText);
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
                showNotification('設定のロードに失敗しました。', 'error');
            },
            complete: function() {
                // ロードボタンを元に戻す
                loadingBtn.html('<i class="fas fa-file mr-2"></i>ロード');
            }
        });
    });
 
    // フォームフィールドの更新
    function updateFormFields(data) {
        console.log("Received data for form update:", data);
 
        // 日付の設定
        $('#start_date').val(data.start_date);
        $('#end_date').val(data.end_date);
        
        // インデックスの設定
        $('input[name="indices"]').prop('checked', false);
        data.selected_indices.forEach(index => {
            $(`input[name="indices"][value="${index}"]`).prop('checked', true);
        });
        
        // コストの設定
        const assetNames = ['SP500', 'Nikkei', 'EuroStoxx', 'FTSE', 'DowJones', 'AUS200', 'HK50', 'SMI20', 'VIX'];
        const spreads = data.spreads.map(Number);
        const swapsLong = data.swaps_long.map(Number);
        const swapsShort = data.swaps_short.map(Number);
        
        assetNames.forEach((asset, index) => {
            if (index < spreads.length) {
                $(`input[name="spread_${asset}"]`).val(spreads[index]);
                $(`input[name="swap_long_${asset}"]`).val(swapsLong[index]);
                $(`input[name="swap_short_${asset}"]`).val(swapsShort[index]);
            }
        });
 
        // レバレッジと契約サイズの設定
        const leverage = data.leverage;
        const contract_size = data.contract_size;
        for (const asset in leverage) {
            $(`input[name="leverage_${asset}"]`).val(leverage[asset]);
        }
        for (const asset in contract_size) {
            $(`input[name="contract_size_${asset}"]`).val(contract_size[asset]);
        }
    
        // その他の設定
        $('#initial_margin').val(data.initial_margin);
        $('#min_lot').val(data.min_lot);
        $('#max_lot').val(data.max_lot);
        $('#handle_small_hedge').val(data.handle_small_hedge);
        $('#settlement_frequency').val(data.settlement_frequency);
        $('#optimize_weights').prop('checked', data.optimize_weights);
        $('#setting_name').val(data.name);
 
        // 更新された値をログ出力
        console.log("Updated form values:");
        console.log("Initial Margin:", $('#initial_margin').val());
        console.log("Min Lot:", $('#min_lot').val());
        console.log("Max Lot:", $('#max_lot').val());
        console.log("Handle Small Hedge:", $('#handle_small_hedge').val());
        console.log("Settlement Frequency:", $('#settlement_frequency').val());
 
        // 資産ごとの値をログ出力
        assetNames.forEach(asset => {
            console.log(`${asset} values:`);
            console.log(`  Spread: ${$(`input[name="spread_${asset}"]`).val()}`);
            console.log(`  Swap Long: ${$(`input[name="swap_long_${asset}"]`).val()}`);
            console.log(`  Swap Short: ${$(`input[name="swap_short_${asset}"]`).val()}`);
            console.log(`  Leverage: ${$(`input[name="leverage_${asset}"]`).val()}`);
            console.log(`  Contract Size: ${$(`input[name="contract_size_${asset}"]`).val()}`);
        });
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
 
    // ツールチップの初期化
    $('[data-toggle="tooltip"]').tooltip();
 
    // スムーズスクロール
    $('a[href^="#"]').on('click', function(e) {
        e.preventDefault();
        const target = $(this).attr('href');
        if (target.length) {
            $('html, body').animate({
                scrollTop: $(target).offset().top - 20
            }, 500);
        }
    });
 
    // 各ロットサイズのヘッジロットデータを読み込む関数
    function loadHedgeLotsData(lotSize) {
        $.ajax({
            url: `/static/output/lot_${lotSize}/hedge_lots.csv`,
            dataType: "text",
            success: function(data) {
                // CSVデータを解析
                let rows = data.split('\n');
                let headers = rows[0].split(',');
                
                // 最新の5行を取得（ヘッダーを除く）
                let latestRows = rows.slice(-6, -1).reverse();
                
                // テーブルにデータを挿入
                let tableBody = $(`#hedge-lots-${lotSize.toString().replace('.', '-')}`);
                tableBody.empty(); // 既存のデータをクリア
                
                latestRows.forEach(row => {
                    let columns = row.split(',');
                    let tr = $('<tr>');
                    
                    // 日付
                    tr.append($('<td>').text(columns[0]));
                    
                    // 各資産のヘッジロット
                    for (let i = 1; i < columns.length; i++) {
                        let value = parseFloat(columns[i]);
                        tr.append($('<td>').text(isNaN(value) ? '-' : value.toFixed(2)));
                    }
                    
                    tableBody.append(tr);
                });
            },
            error: function(xhr, status, error) {
                console.error(`Error loading hedge lots for lot size ${lotSize}: ${error}`);
                $(`#hedge-lots-${lotSize.toString().replace('.', '-')}`).append(
                    '<tr><td colspan="10" class="text-center text-danger">' +
                    '<i class="fas fa-exclamation-circle mr-2"></i>データの読み込みに失敗しました</td></tr>'
                );
            }
        });
    }
 
    // 全てのロットサイズのデータを読み込む
    if (typeof lot_sizes !== 'undefined') {
        lot_sizes.forEach(lotSize => {
            loadHedgeLotsData(lotSize);
        });
    }
 
// 30秒ごとにデータを更新
setInterval(function() {
    if (typeof lot_sizes !== 'undefined') {
        lot_sizes.forEach(lotSize => {
            loadHedgeLotsData(lotSize);
        });
    }
}, 30000);

// ソケット接続でリアルタイムデータを受信した時の処理
socket.on('hedge_lots_update', function(data) {
    if (data.lot_size) {
        loadHedgeLotsData(data.lot_size);
    }
});

// フォームの変更を監視し、値の変更をログ出力
$('.form-control').on('change', function() {
    const $this = $(this);
    console.log(`Form field changed: ${$this.attr('name')} = ${$this.val()}`);
});

// フォーム送信前の値の検証
$('#mainForm').on('submit', function(e) {
    const formData = new FormData(this);
    console.log('Form submission values:');
    for (const [key, value] of formData.entries()) {
        console.log(`${key}: ${value}`);
    }
});

// ページ読み込み時の初期値をログ出力
$(document).ready(function() {
    console.log('Initial form values:');
    console.log('Initial Margin:', $('#initial_margin').val());
    console.log('Min Lot:', $('#min_lot').val());
    console.log('Max Lot:', $('#max_lot').val());
    console.log('Handle Small Hedge:', $('#handle_small_hedge').val());
    console.log('Settlement Frequency:', $('#settlement_frequency').val());

    const assetNames = ['SP500', 'Nikkei', 'EuroStoxx', 'FTSE', 'DowJones', 'AUS200', 'HK50', 'SMI20', 'VIX'];
    assetNames.forEach(asset => {
        console.log(`${asset}:`);
        console.log(`  spread_${asset}: ${$(`input[name="spread_${asset}"]`).val()}`);
        console.log(`  swap_long_${asset}: ${$(`input[name="swap_long_${asset}"]`).val()}`);
        console.log(`  swap_short_${asset}: ${$(`input[name="swap_short_${asset}"]`).val()}`);
        console.log(`  leverage_${asset}: ${$(`input[name="leverage_${asset}"]`).val()}`);
        console.log(`  contract_size_${asset}: ${$(`input[name="contract_size_${asset}"]`).val()}`);
    });
});
});
