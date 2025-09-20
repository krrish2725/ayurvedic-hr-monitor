$(document).ready(function() {
    // DOM Elements
    const $startBtn = $('#start-btn');
    const $stopBtn = $('#stop-btn');
    const $heartRateDisplay = $('#heart-rate');
    const $cameraFeed = $('#camera-feed');  // Camera feed element
    const $ppgPlot = $('#ppg-plot');
    const $plotContainer = $('#plot-container');
    const $ayurvedicInsights = $('#ayurvedic-insights');
    const $doshaBadge = $('#dosha-badge');
    const $doshaDescription = $('#dosha-description');
    const $recommendations = $('#recommendations');
    const $instructionModal = $('#instruction-modal');
    const $instructionText = $('#instruction-text');

    // Measurement state
    let isMeasuring = false;
    let measurementInterval;
    let measurementStartTime = 0;
    const MEASUREMENT_DURATION = 20000; // 20 seconds

    // Liveness check timeout
    let livenessCheckTimeout;
    const LIVENESS_CHECK_TIMEOUT = 30000; // 30 seconds

    // Start measurement
    $startBtn.on('click', function() {
        if (isMeasuring) return;
        
        console.log('Start measurement button clicked');
        
        // Reset UI
        $heartRateDisplay.text('--');
        $ppgPlot.addClass('d-none');
        $plotContainer.find('p').removeClass('d-none');
        $ayurvedicInsights.addClass('d-none');
        
        // Show loading state
        $startBtn.prop('disabled', true).html('<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Starting...');
        
        // Add pulsing effect to camera feed
        $cameraFeed.addClass('measurement-active');
        
        // Start liveness check timeout
        livenessCheckTimeout = setTimeout(function() {
            if (isMeasuring) {
                showError('Liveness check timed out. Please ensure your face is well-lit and visible.');
                stopMeasurement();
            }
        }, LIVENESS_CHECK_TIMEOUT);
        
        // Start measurement on the server
        console.log('Sending request to /start_measurement');
        $.ajax({
            url: '/start_measurement',
            type: 'POST',
            dataType: 'json',
            method: 'POST',
            success: function(response) {
                console.log('Start measurement response:', response);
                if (response.status === 'success') {
                    isMeasuring = true;
                    $startBtn.prop('disabled', true);
                    $stopBtn.prop('disabled', false);
                    $startBtn.html('Measuring...');
                    
                    // Show liveness check instructions
                    showInstruction('Please look at the camera and blink your eyes to verify liveness');
                    
                    // Start polling for updates
                    startPolling();
                } else {
                    console.error('Error in response:', response.message);
                    showError(response.message || 'Failed to start measurement');
                    resetUI();
                }
            },
            error: function(xhr, status, error) {
                console.error('AJAX error:', status, error);
                clearTimeout(livenessCheckTimeout);
                showError(`Failed to start measurement: ${status} - ${error}`);
                resetUI();
            }
        });
    });
    
    // Stop measurement button handler
    $stopBtn.on('click', stopMeasurement);
    
    // Start polling for updates and manage measurement timeout
    function startPolling() {
        // Reset the measurement start time
        measurementStartTime = Date.now();
        
        // Clear any existing interval
        clearInterval(measurementInterval);
        
        // Start a new interval to check the elapsed time
        measurementInterval = setInterval(function() {
            if (!isMeasuring) return;
            
            const elapsed = Date.now() - measurementStartTime;
            const remaining = Math.max(0, MEASUREMENT_DURATION - elapsed);
            
            // Update the button text with remaining time
            const seconds = Math.ceil(remaining / 1000);
            $startBtn.html(`Measuring... (${seconds}s)`);
            
            // Auto-stop when time is up
            if (remaining <= 0) {
                stopMeasurement();
            }
        }, 200); // Update every 200ms for smoother countdown
    }
    
    // Display results
    function displayResults(data) {
        console.log('Displaying results:', data); // Debug log
        if (data.status === 'Error') {
            showError(data.message);
            return;
        }
        
        // Ensure we have valid data
        if (data.heart_rate === undefined || data.heart_rate === null) {
            showError('No heart rate data available');
            return;
        }
        
        // Update heart rate - ensure it's a clean number
        const heartRate = Math.round(parseFloat(data.heart_rate) * 10) / 10; // Round to 1 decimal place
        $heartRateDisplay.text(heartRate.toFixed(0)); // Display as integer
        
        // Show PPG plot
        $plotContainer.find('p').addClass('d-none');
        if (data.plot) {
            $ppgPlot.attr('src', 'data:image/png;base64,' + data.plot);
            $ppgPlot.removeClass('d-none');
        }
        
        // Update Ayurvedic insights
        updateAyurvedicInsights(data.dosha, data.ayurvedic_info);
    }
    
    // Update Ayurvedic insights
    function updateAyurvedicInsights(dosha, info) {
        $doshaBadge.html(`<span class="dosha-badge ${dosha}">${dosha.toUpperCase()} DOSHA</span>`);
        $doshaDescription.text(info.description);
        
        // Update recommendations
        $recommendations.empty();
        info.recommendations.forEach(function(rec) {
            $recommendations.append(`<li class="list-group-item">${rec}</li>`);
        });
        
        // Show insights
        $ayurvedicInsights.removeClass('d-none');
    }
    
    // Stop the current measurement
    function stopMeasurement() {
        if (!isMeasuring) return;
        
        clearInterval(measurementInterval);
        clearTimeout(livenessCheckTimeout);
        
        // Disable buttons and show loading state
        $stopBtn.prop('disabled', true).html('<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Stopping...');
        
        // Show processing message
        const processingMsg = 'Processing your heart rate data...';
        showInstruction(processingMsg);
        
        // Stop measurement on the server
        $.ajax({
            url: '/stop_measurement',
            method: 'POST',
            timeout: 10000, // 10 second timeout
            success: function(response) {
                hideInstruction();
                if (response.status === 'Warning' || response.status === 'Error') {
                    showError(response.message);
                } else {
                    displayResults(response);
                }
                resetUI();
            },
            error: function(xhr, status, error) {
                hideInstruction();
                if (status === 'timeout') {
                    showError('Processing is taking longer than expected. Please try again.');
                } else {
                    showError('Failed to process measurement. Please try again.');
                }
                resetUI();
            }
        });
    }
    
    // Reset UI to initial state
    function resetUI() {
        isMeasuring = false;
        measurementStartTime = 0;
        $startBtn.prop('disabled', false).text('Start Measurement');
        $stopBtn.prop('disabled', true).text('Stop Measurement');
        $cameraFeed.removeClass('measurement-active');
        clearInterval(measurementInterval);
        clearTimeout(livenessCheckTimeout);
        // Don't hide instruction here as we might want to show processing message
    }
    
    // Show instruction message
    function showInstruction(message) {
        hideInstruction();
        const $instruction = $('<div class="alert alert-info instruction-alert">' + message + '</div>');
        $('body').append($instruction);
        setTimeout(() => $instruction.addClass('show'), 100);
    }
    
    // Hide instruction message
    function hideInstruction() {
        $('.instruction-alert').remove();
    }
    
    // Show error message
    function showError(message) {
        hideInstruction();
        const $error = $('<div class="alert alert-danger error-alert">' + message + '</div>');
        $('body').append($error);
        setTimeout(() => $error.addClass('show'), 100);
        setTimeout(() => {
            $error.removeClass('show');
            setTimeout(() => $error.remove(), 300);
        }, 5000);
    }

    // Function to update the video feed with a timestamp to prevent caching
    function updateVideoFeed() {
        const timestamp = new Date().getTime();
        $cameraFeed.attr('src', '/video_feed?' + timestamp);
    }

    // Initial video feed update
    updateVideoFeed();
    
    // Refresh the video feed every 30 seconds to prevent timeouts
    setInterval(updateVideoFeed, 30000);
    
    // Add error handling for the image
    $cameraFeed.on('error', function() {
        console.error('Error loading video feed');
        // Try to reload the feed after a delay
        setTimeout(updateVideoFeed, 1000);
    });
});
