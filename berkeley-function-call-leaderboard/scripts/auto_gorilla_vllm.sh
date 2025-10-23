#!/bin/bash

# Comprehensive model evaluation script  
# This script starts vLLM for each model and runs evaluations on BFCL v3 benchmark

# Configuration
VLLM_PORT=8025
BFCL_DIR="/home/aleksanderk/mlf2/work/experimental/gorilla/berkeley-function-call-leaderboard"
RESULTS_DIR="$BFCL_DIR/result"
SCORE_DIR="$BFCL_DIR/score"
LOG_DIR="./evaluation_logs"
EVALUATION_TIMEOUT=7200  # 2 hours timeout for each evaluation
VLLM_STARTUP_TIMEOUT=300  # 5 minutes for vLLM to start
TEMPERATURE=0.0  # Temperature for BFCL evaluation

declare -a TASK_ACCURACY_RESULTS=()

# Utility to extract accuracy metrics from score JSON
extract_accuracy_summary() {
    local score_file="$1"
    python3 - "$score_file" <<'PY'
import json
import sys

file_path = sys.argv[1]

try:
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
except Exception:
    sys.exit(1)

data = None
for line in lines:
    try:
        candidate = json.loads(line)
    except json.JSONDecodeError:
        continue
    if isinstance(candidate, dict) and "accuracy" in candidate:
        data = candidate
        break
    if isinstance(candidate, list):
        for item in candidate:
            if isinstance(item, dict) and "accuracy" in item:
                data = item
                break
        if data:
            break

if not data:
    sys.exit(2)

accuracy = data.get("accuracy")
if accuracy is None:
    sys.exit(3)

correct = data.get("correct_count")
total = data.get("total_count")

def fmt(value):
    if value is None:
        return ""
    return str(value)

print(f"{accuracy * 100:.2f}|{fmt(correct)}|{fmt(total)}")
PY
}

# Set environment variables for BFCL
export VLLM_ENDPOINT="localhost"
export VLLM_PORT="$VLLM_PORT"
export BFCL_PROJECT_ROOT="$BFCL_DIR"
MAX_RESTART_ATTEMPTS=15 # Maximum number of restart attempts per evaluation
VLLM_HEALTH_CHECK_INTERVAL=4  # Check vLLM health every N monitoring cycles (N * 30 seconds)
VLLM_PID=""
VLLM_PID_FILE="$PWD/vllm_pid.txt"

SCRIPT_PID=$$
VLLM_SESSION_ID=""

# Create directories
mkdir -p "$RESULTS_DIR" "$SCORE_DIR" "$LOG_DIR"

# Cleanup function
cleanup() {
    echo "$(date): Cleaning up..."
    if [ ! -z "$VLLM_PID" ] && ps -p $VLLM_PID > /dev/null 2>&1; then
        echo "$(date): Stopping vLLM process $VLLM_PID"
        kill -TERM $VLLM_PID 2>/dev/null || true
        sleep 5
    fi
    # Kill any remaining vLLM processes
    kill -9 $VLLM_PID  2>/dev/null || true

    if [ ! -z "$VLLM_SESSION_ID" ]; then
        echo "$(date): Killing processes with session ID $VLLM_SESSION_ID..."
        pkill -f "VLLM_SESSION_ID=$VLLM_SESSION_ID" 2>/dev/null || true
        sleep 5
        pkill -9 -f "VLLM_SESSION_ID=$VLLM_SESSION_ID" 2>/dev/null || true
    fi
    # Kill any remaining evaluation processes
    pkill -f "evalplus.evaluate" 2>/dev/null || true
    sleep 2
}

# Final cleanup function for script termination
final_cleanup() {
    echo "$(date): Final cleanup and exit..."
    cleanup
    exit 0
}

# Set up signal handlers
trap final_cleanup SIGTERM SIGINT EXIT

# Model configurations: model_name:model_path:tensor_parallel_size:max_model_len:gpu_memory_util:extra_args:model_type
# model_type can be: gpt-oss or qwen

MODELS=(
    #"Qwen3-30B-A3B-Coder-Instruct:/mlf8-shared/hf_cache/hub/Qwen3-Coder-30B-A3B-Instruct:4:65536:0.9:--enable-chunked-prefill --trust-remote-code --enable-expert-parallel --rope-scaling '{\"rope_type\":\"yarn\",\"factor\":2.0,\"original_max_position_embeddings\":32768}':qwen"
    #"Qwen3-30B-A3B-Coder-FC:/mlf8-shared/hf_cache/hub/Qwen3-Coder-30B-A3B-Instruct:4:65536:0.9:--enable-chunked-prefill --trust-remote-code --enable-expert-parallel --rope-scaling '{\"rope_type\":\"yarn\",\"factor\":2.0,\"original_max_position_embeddings\":32768}':qwen"
    #"Qwen3-30B-A3B-Coder-FC:/mlf8-shared/aleksanderk/Qwen3-Coder-30B-A3B-gs18x9:4:65536:0.9:--enable-chunked-prefill --trust-remote-code --enable-expert-parallel --rope-scaling '{\"rope_type\":\"yarn\",\"factor\":2.0,\"original_max_position_embeddings\":32768}':qwen"
    #"Qwen3-30B-A3B-Coder-FC:/mlf8-shared/aleksanderk/Qwen3-Coder-30B-A3B-test-18x9:4:65536:0.9:--enable-chunked-prefill --trust-remote-code --enable-expert-parallel --rope-scaling '{\"rope_type\":\"yarn\",\"factor\":2.0,\"original_max_position_embeddings\":32768}':qwen"
    "Qwen3-30B-A3B-Coder-FC:/mlf8-shared/aleksanderk/Qwen3-Coder-30B-A3B-lut-test:4:65536:0.9:--enable-chunked-prefill --trust-remote-code --enable-expert-parallel:qwen"
    #"GPT-OSS-120B-FC:/mlf6-shared/aleksanderk/gpt-oss-120b-lut:4:65536:0.9:--enable-chunked-prefill --trust-remote-code --enable-expert-parallel:gpt-oss"
    #"GPT-OSS-120B-FC:/mlf8-shared/hf_cache/hub/gpt-oss-120b-bf16:4:65536:0.9:--enable-chunked-prefill --trust-remote-code --enable-expert-parallel:gpt-oss"
)

# BFCL test categories to evaluate
# Using main categories for comprehensive evaluation
TEST_CATEGORIES=("simple" "parallel" "multiple" "multi_turn_base") #"simple" "parallel" "multiple" "multi_turn_base"

# Function to check if vLLM is ready
check_vllm_ready() {
    local max_attempts=60  # 10 minutes with 10-second intervals
    local attempt=0
    
    echo "$(date): Checking if vLLM is ready on port $VLLM_PORT..."
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
            echo "$(date): vLLM is ready!"
            return 0
        fi
        
        # Check if vLLM process is still running
        if [ ! -z "$VLLM_PID" ] && ! ps -p $VLLM_PID > /dev/null 2>&1; then
            echo "$(date): ERROR: vLLM process died during startup"
            return 1
        fi
        
        echo "$(date): Waiting for vLLM to be ready... (attempt $((attempt+1))/$max_attempts)"
        sleep 10
        ((attempt++))
    done
    
    echo "$(date): ERROR: vLLM failed to become ready within timeout"
    return 1
}

# Function to monitor vLLM health during evaluation
monitor_vllm_health() {
    local model_name="$1"
    
    # Check if vLLM process is still running
    if [ ! -z "$VLLM_PID" ] && ! ps -p $VLLM_PID > /dev/null 2>&1; then
        echo "$(date): ⚠️  vLLM process died! PID $VLLM_PID no longer exists"
        return 1
    fi
    
    # Check if vLLM API is still responding
    if ! curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
        echo "$(date): ⚠️  vLLM API not responding on port $VLLM_PORT"
        return 1
    fi
    
    return 0
}

# Function to restart vLLM if it fails
restart_vllm_if_needed() {
    local model_name="$1"
    local model_path="$2"
    local tp_size="$3"
    local max_len="$4"
    local gpu_util="$5"
    local extra_args="$6"
    local max_restart_attempts=3
    local restart_attempt=1
    
    while [ $restart_attempt -le $max_restart_attempts ]; do
        if monitor_vllm_health "$model_name"; then
            return 0  # vLLM is healthy
        fi
        
        echo "$(date): vLLM health check failed (attempt $restart_attempt/$max_restart_attempts)"
        echo "$(date): Attempting to restart vLLM for $model_name..."
        
        # Clean up existing vLLM processes
        if [ ! -z "$VLLM_PID" ]; then
            echo "$(date): Killing existing vLLM process $VLLM_PID"
            kill -9 $VLLM_PID 2>/dev/null || true
        fi
        kill -11 $VLLM_PID 2>/dev/null || true
        sleep 10
        
        # Restart vLLM
        if start_vllm "$model_name" "$model_path" "$tp_size" "$max_len" "$gpu_util" "$extra_args"; then
            echo "$(date): ✅ vLLM successfully restarted for $model_name"
            return 0
        else
            echo "$(date): ❌ Failed to restart vLLM (attempt $restart_attempt/$max_restart_attempts)"
            ((restart_attempt++))
            if [ $restart_attempt -le $max_restart_attempts ]; then
                echo "$(date): Waiting 30 seconds before next restart attempt..."
                sleep 30
            fi
        fi
    done
    
    echo "$(date): ❌ Failed to restart vLLM after $max_restart_attempts attempts"
    return 1
}

# Function to start vLLM
start_vllm() {
    local model_name="$1"
    local model_path="$2"
    local tp_size="$3"
    local max_len="$4"
    local gpu_util="$5"
    local extra_args="$6"
    
    echo "$(date): Starting vLLM for model: $model_name"
    echo "$(date): Model path: $model_path"
    echo "$(date): Config: TP=$tp_size, MaxLen=$max_len, GPU_Util=$gpu_util"
    
    local log_file="$LOG_DIR/vllm_${model_name}.log"
    

     # Create a unique session identifier for this script instance
    local session_id="simpleeval_$$_$(date +%s)"
    served_model_name="${model_name%-FC}"
    
    # Build vLLM command with unique identifier
    local vllm_cmd="vllm serve '$model_path' \
        --gpu-memory-utilization $gpu_util \
        --max-model-len=$max_len \
        --tensor-parallel-size $tp_size \
        --served-model-name $served_model_name \
        --port $VLLM_PORT \
        $extra_args"
    
    echo "$(date): Running: $vllm_cmd"
    echo "$(date): Log file: $log_file"
    echo "$(date): Session ID: $session_id"
    
    # Start vLLM in a new process group with unique identifier
    setsid bash -c "
        export VLLM_SESSION_ID='$session_id'
        exec $vllm_cmd
    " > "$log_file" 2>&1 &
    
    VLLM_PID=$!
    VLLM_CMD="$vllm_cmd"
    VLLM_SESSION_ID="$session_id"

    echo $VLLM_PID > $VLLM_PID_FILE
    echo "$VLLM_SESSION_ID" > "$PWD/vllm_session.txt"
    
    echo "$(date): vLLM started with PID: $VLLM_PID, Session: $session_id"
    
    # Wait for vLLM to be ready
    if check_vllm_ready; then
        echo "$(date): vLLM successfully started for $model_name"
        return 0
    else
        echo "$(date): Failed to start vLLM for $model_name"
        return 1
    fi
}

# Function to check if evaluation is making progress
check_evaluation_progress() {
    local eval_log="$1"
    local eval_pid="$2"
    local last_progress_time="$3"
    
    # Check if process is still running
    if ! ps -p $eval_pid > /dev/null 2>&1; then
        return 1  # Process died
    fi
    
    # Check if log file is being updated
    if [ -f "$eval_log" ]; then
        local last_update=$(stat -c %Y "$eval_log" 2>/dev/null || echo 0)
        local current_time=$(date +%s)
        local time_since_update=$((current_time - last_update))
        local time_since_progress=$((current_time - last_progress_time))
        
        # Consider it hung if no log updates for 15 minutes OR no progress for 30 minutes
        if [ $time_since_update -gt 900 ] || [ $time_since_progress -gt 1800 ]; then
            return 2  # Evaluation appears hung
        fi
        
        # Check if there's actual progress in the log (look for recent activity)
        if [ $time_since_update -lt 60 ]; then
            # Recent log activity, update progress time
            echo $current_time
            return 0
        fi
    fi
    
    return 0  # Still running, no issues detected
}

# Function to run BFCL evaluation with monitoring and restart capability
run_evaluation_with_monitoring() {
    local model_name="$1"
    local test_category="$2"
    local model_path="$3"
    local tp_size="$4"
    local max_len="$5"
    local gpu_util="$6"
    local extra_args="$7"
    local model_type="$8"
    
    echo "$(date): Starting BFCL evaluation: $model_name on $test_category (model_type: $model_type)"
    
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local eval_log="$LOG_DIR/eval_${model_name}_${test_category}_${timestamp}.log"
    local attempt=1
    
    while [ $attempt -le $MAX_RESTART_ATTEMPTS ]; do
        echo "$(date): Evaluation attempt $attempt/$MAX_RESTART_ATTEMPTS"
        
        # Build BFCL generate command based on model type
        local eval_cmd
        if [ "$model_type" = "gpt-oss" ]; then
            # GPT-OSS models use OpenAI API compatible interface
            eval_cmd="cd '$BFCL_DIR' && OPENAI_BASE_URL='http://localhost:$VLLM_PORT/v1' OPENAI_API_KEY='fake' BFCL_PROJECT_ROOT='$BFCL_DIR' bfcl generate \
                --model '$model_name' \
                --test-category '$test_category' \
                --temperature $TEMPERATURE \
                --allow-overwrite \
                --local-model-path '$model_path' \
                --num-threads 8"
        else
            # Qwen models use vLLM backend
            eval_cmd="cd '$BFCL_DIR' && bfcl generate \
                --model '$model_name' \
                --test-category '$test_category' \
                --backend vllm \
                --skip-server-setup \
                --temperature $TEMPERATURE \
                --allow-overwrite \
                --local-model-path '$model_path'"
        fi
        
        echo "$(date): BFCL command: $eval_cmd"
        echo "$(date): Results will be saved to: $RESULTS_DIR/$model_name/"
        echo "$(date): Evaluation log: $eval_log"
        
        # Start evaluation in background
        bash -c "$eval_cmd" > "$eval_log" 2>&1 &
        local eval_pid=$!
        echo "$(date): Evaluation started with PID: $eval_pid"
        
        # Monitor evaluation process
        local monitor_count=0
        local last_progress_time=$(date +%s)
        local evaluation_hung=false
        local vllm_failed=false
        
        while ps -p $eval_pid > /dev/null 2>&1; do
            sleep 30
            ((monitor_count++))
            
            # Check vLLM health periodically
            if [ $((monitor_count % VLLM_HEALTH_CHECK_INTERVAL)) -eq 0 ]; then
                if ! monitor_vllm_health "$model_name"; then
                    echo "$(date): ⚠️  vLLM health check failed during evaluation"
                    # Try to restart vLLM
                    if restart_vllm_if_needed "$model_name" "$model_path" "$tp_size" "$max_len" "$gpu_util" "$extra_args"; then
                        echo "$(date): ✅ vLLM restarted successfully, evaluation may continue"
                    else
                        echo "$(date): ❌ Failed to restart vLLM, terminating evaluation"
                        kill -9 $eval_pid 2>/dev/null || true
                        vllm_failed=true
                        break
                    fi
                fi
            fi
            
            # Check progress and detect hangs
            local progress_check=$(check_evaluation_progress "$eval_log" "$eval_pid" "$last_progress_time")
            case $? in
                0)  # Still running normally
                    if [ "$progress_check" != "$last_progress_time" ]; then
                        last_progress_time=$progress_check
                    fi
                    ;;
                1)  # Process died
                    echo "$(date): Evaluation process died unexpectedly"
                    break
                    ;;
                2)  # Evaluation appears hung
                    echo "$(date): Evaluation appears to be hung, killing process..."
                    kill -9 $eval_pid 2>/dev/null || true
                    sleep 5
                    evaluation_hung=true
                    break
                    ;;
            esac
            
            # Regular progress reporting every 5 minutes
            if [ $((monitor_count % 10)) -eq 0 ]; then
                echo "$(date): BFCL evaluation still running for $model_name on $test_category (${monitor_count}0 seconds elapsed)"
                # Show last few lines of log for progress indication
                if [ -f "$eval_log" ]; then
                    echo "$(date): Recent log activity:"
                    tail -3 "$eval_log" 2>/dev/null || echo "No recent log output"
                fi
            fi
            
            # Hard timeout check
            if [ $((monitor_count * 30)) -gt $EVALUATION_TIMEOUT ]; then
                echo "$(date): Evaluation reached hard timeout, killing process..."
                kill -9 $eval_pid 2>/dev/null || true
                sleep 5
                evaluation_hung=true
                break
            fi
        done
        
        # Check evaluation result
        wait $eval_pid 2>/dev/null
        local exit_code=$?
        
        if [ $exit_code -eq 0 ] && [ "$evaluation_hung" = false ] && [ "$vllm_failed" = false ]; then
            echo "$(date): ✅ BFCL generation completed successfully: $model_name on $test_category"
            
            # Check if results were generated
            local result_file="$RESULTS_DIR/$model_name/BFCL_v3_${test_category}_result.json"
            if [ -f "$result_file" ]; then
                echo "$(date): Results saved to: $result_file"
                
                # Run BFCL evaluation to get scores
                echo "$(date): Running BFCL evaluation to compute scores..."
                local eval_score_cmd
                if [ "$model_type" = "gpt-oss" ]; then
                    eval_score_cmd="cd '$BFCL_DIR' && OPENAI_API_KEY='fake' bfcl evaluate --model '$model_name' --test-category '$test_category'"
                else
                    eval_score_cmd="cd '$BFCL_DIR' && bfcl evaluate --model '$model_name' --test-category '$test_category'"
                fi
                echo "$(date): Score command: $eval_score_cmd"
                
                # Run evaluation with timeout
                timeout 300 bash -c "$eval_score_cmd" >> "$eval_log" 2>&1
                local score_exit_code=$?
                
                if [ $score_exit_code -eq 0 ]; then
                    local score_file="$SCORE_DIR/$model_name/BFCL_v3_${test_category}_score.json"
                    if [ -f "$score_file" ]; then
                        echo "$(date): ✅ Scores computed and saved to: $score_file"

                        local metrics accuracy_percent_fmt correct_count total_count
                        if metrics=$(extract_accuracy_summary "$score_file"); then
                            IFS='|' read -r accuracy_percent_fmt correct_count total_count <<< "$metrics"
                            correct_count=${correct_count:-N/A}
                            total_count=${total_count:-N/A}
                            echo "$(date): Accuracy for $model_name on $test_category: ${accuracy_percent_fmt}% (${correct_count}/${total_count})"
                            TASK_ACCURACY_RESULTS+=("$model_name|$test_category|$accuracy_percent_fmt|$correct_count|$total_count")
                        else
                            echo "$(date): ⚠️  WARNING: Unable to parse accuracy from $score_file"
                        fi
                    else
                        echo "$(date): ⚠️  WARNING: Score computation completed but no score file found"
                    fi
                elif [ $score_exit_code -eq 124 ]; then
                    echo "$(date): ⚠️  WARNING: Score computation timed out after 5 minutes, but generation succeeded"
                else
                    echo "$(date): ⚠️  WARNING: Score computation failed (exit code: $score_exit_code), but generation succeeded"
                fi
                
                return 0  # Success
            else
                echo "$(date): ⚠️  WARNING: No results found at $result_file"
            fi
        fi
        
        # If we get here, the evaluation failed or hung
        if [ "$vllm_failed" = true ]; then
            echo "$(date): ❌ BFCL evaluation failed due to vLLM failure on attempt $attempt"
        elif [ "$evaluation_hung" = true ]; then
            echo "$(date): ❌ BFCL evaluation hung on attempt $attempt"
        else
            echo "$(date): ❌ BFCL evaluation failed with exit code $exit_code on attempt $attempt"
        fi
        
        # Clean up any remaining processes
        pkill -f "bfcl generate" 2>/dev/null || true
        sleep 5
        
        if [ $attempt -lt $MAX_RESTART_ATTEMPTS ]; then
            echo "$(date): Retrying evaluation in 30 seconds..."
            sleep 30
            # Update log file name for next attempt
            eval_log="$LOG_DIR/eval_${model_name}_${test_category}_${timestamp}_attempt${attempt}.log"
        fi
        
        ((attempt++))
    done
    
    echo "$(date): ❌ All BFCL evaluation attempts failed for $model_name on $test_category"
    return 1
}

# Main execution
main() {
    echo "$(date): Starting automated BFCL model evaluation"
    echo "$(date): Models to evaluate: ${#MODELS[@]}"
    echo "$(date): Test categories: ${TEST_CATEGORIES[*]}"
    echo "$(date): Temperature: $TEMPERATURE"
    echo "$(date): Max restart attempts: $MAX_RESTART_ATTEMPTS"
    echo "$(date): vLLM health check interval: every $((VLLM_HEALTH_CHECK_INTERVAL * 30)) seconds"
    echo "$(date): Results directory: $RESULTS_DIR"
    echo "$(date): Score directory: $SCORE_DIR"
    echo "$(date): Logs directory: $LOG_DIR"
    
    local total_evaluations=0
    local successful_evaluations=0

    # Kill vLLM if it is running
    if [ -f "$VLLM_PID_FILE" ]; then
        VLLM_PID=$(cat $VLLM_PID_FILE)
        kill -11 $VLLM_PID 2>/dev/null || true
        rm $VLLM_PID_FILE
        VLLM_PID=""
    fi
    sleep 5
    
    # Process each model
    for model_config in "${MODELS[@]}"; do
        # Parse model configuration
        IFS=':' read -r model_name model_path tp_size max_len gpu_util extra_args model_type <<< "$model_config"
        
        # Default to qwen if model_type is not specified
        if [ -z "$model_type" ]; then
            model_type="qwen"
        fi
        
        echo ""
        echo "$(date): =================================================="
        echo "$(date): Starting evaluation for model: $model_name (type: $model_type)"
        echo "$(date): =================================================="
        
        # Start vLLM for this model
        if restart_vllm_if_needed "$model_name" "$model_path" "$tp_size" "$max_len" "$gpu_util" "$extra_args"; then
            # Run evaluations on all test categories
            for test_category in "${TEST_CATEGORIES[@]}"; do
                echo ""
                echo "$(date): --------------------------------------------------"
                echo "$(date): Evaluating $model_name on $test_category"
                echo "$(date): --------------------------------------------------"
                
                if run_evaluation_with_monitoring "$model_name" "$test_category" "$model_path" "$tp_size" "$max_len" "$gpu_util" "$extra_args" "$model_type"; then
                    ((successful_evaluations++))
                    echo "$(date): ✅ Successfully completed: $model_name on $test_category"
                else
                    echo "$(date): ❌ Failed: $model_name on $test_category"
                fi

                ((total_evaluations++))
            done
        else
            echo "$(date): ❌ Failed to start vLLM for $model_name, skipping evaluations"
            total_evaluations=$((total_evaluations + ${#TEST_CATEGORIES[@]}))
        fi
        
        # Stop vLLM before moving to next model
        cleanup
        VLLM_PID=""
        
        echo "$(date): Waiting 30 seconds before next model..."
        sleep 30
    done
    
    # Final summary
    echo ""
    echo "$(date): =================================================="
    echo "$(date): BFCL EVALUATION COMPLETE"
    echo "$(date): =================================================="
    echo "$(date): Total evaluations: $total_evaluations"
    echo "$(date): Successful evaluations: $successful_evaluations"
    echo "$(date): Success rate: $(( successful_evaluations * 100 / total_evaluations ))%"
    echo "$(date): Results saved in: $RESULTS_DIR"
    echo "$(date): Scores saved in: $SCORE_DIR"
    echo "$(date): Logs saved in: $LOG_DIR"

    if [ ${#TASK_ACCURACY_RESULTS[@]} -gt 0 ]; then
        echo "$(date): --------------------------------------------------"
        echo "$(date): Accuracy summary by task"
        for record in "${TASK_ACCURACY_RESULTS[@]}"; do
            IFS='|' read -r summary_model summary_category summary_accuracy summary_correct summary_total <<< "$record"
            echo "$(date):   - $summary_model / $summary_category: ${summary_accuracy}% (${summary_correct}/${summary_total})"
        done
    fi
    
    # Generate overall summary
    local summary_file="$RESULTS_DIR/overall_summary_$(date +%Y%m%d_%H%M%S).txt"
    {
        echo "BFCL Model Evaluation Summary"
        echo "============================"
        echo "Date: $(date)"
        echo "Total models: ${#MODELS[@]}"
        echo "Total test categories: ${#TEST_CATEGORIES[@]}"
        echo "Temperature: $TEMPERATURE"
        echo "Max restart attempts: $MAX_RESTART_ATTEMPTS"
        echo "Total evaluations: $total_evaluations"
        echo "Successful evaluations: $successful_evaluations"
        echo "Success rate: $(( successful_evaluations * 100 / total_evaluations ))%"
        echo ""
        echo "Models evaluated:"
        for model_config in "${MODELS[@]}"; do
            IFS=':' read -r model_name model_path _ _ _ _ <<< "$model_config"
            echo "  - $model_name ($model_path)"
        done
        echo ""
        echo "Test categories: ${TEST_CATEGORIES[*]}"
        echo ""
        echo "Results directory: $RESULTS_DIR"
        echo "Score directory: $SCORE_DIR"
        echo "Logs directory: $LOG_DIR"

        if [ ${#TASK_ACCURACY_RESULTS[@]} -gt 0 ]; then
            echo ""
            echo "Task accuracy results:"
            for record in "${TASK_ACCURACY_RESULTS[@]}"; do
                IFS='|' read -r summary_model summary_category summary_accuracy summary_correct summary_total <<< "$record"
                echo "  - $summary_model / $summary_category: ${summary_accuracy}% (${summary_correct}/${summary_total})"
            done
        fi
    } > "$summary_file"
    
    echo "$(date): Summary saved to: $summary_file"
}

# Check dependencies
if ! command -v vllm &> /dev/null; then
    echo "ERROR: vllm command not found. Please install vLLM."
    exit 1
fi

if ! command -v curl &> /dev/null; then
    echo "ERROR: curl command not found. Please install curl."
    exit 1
fi

if ! command -v bfcl &> /dev/null; then
    echo "ERROR: bfcl command not found. Please install BFCL."
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 command not found. Please install Python 3."
    exit 1
fi

if [ ! -d "$BFCL_DIR" ]; then
    echo "ERROR: BFCL directory not found: $BFCL_DIR"
    exit 1
fi

# Start main execution
main "$@"
