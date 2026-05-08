# PowerShell script to automate training, sampling, and evaluation

# Define configurations
$batchSizes = @(4, 6, 8)  # Example batch sizes
$learningRates = @(0.0001, 0.0005)  # Example learning rates

# Define other parameters
$trainingSteps = 1000
$saveInterval = 100
$logInterval = 25
$dataDir = "datasets"
$imageDir = "datasets/slake/imgs"
$dataset = "slake"

# Define paths
$pythonPath = ".\venv\Scripts\python.exe"
$trainScript = ".\train.py"
$sampleScript = ".\sample_vqa_GPU.py"
$evalScript = ".\eval_DiffuVQA.py"

# Iterate over configurations
foreach ($batchSize in $batchSizes) {
    foreach ($lr in $learningRates) {
        # Create a unique checkpoint path for this configuration
        $checkpointPath = "checkpoints/batch${batchSize}_lr${lr}"
        New-Item -ItemType Directory -Force -Path $checkpointPath

        # Train the model
        Write-Host "Training with batch size $batchSize and learning rate $lr..."
        & $pythonPath $trainScript `
            --lr $lr `
            --batch_size $batchSize `
            --learning_steps $trainingSteps `
            --save_interval $saveInterval `
            --log_interval $logInterval `
            --data_dir $dataDir `
            --image_dir $imageDir `
            --dataset $dataset `
            --checkpoint_path $checkpointPath

        # Sample and evaluate at each saved checkpoint
        for ($step = $saveInterval; $step -le $trainingSteps; $step += $saveInterval) {
            $formattedStep = $step.ToString("000000")
            $checkpointFile = "${checkpointPath}/ema_0.9999_${formattedStep}.pt"

            # Sample
            Write-Host "Sampling from checkpoint $checkpointFile..."
            # Iterate over step values from 5 to 20 with increments of 5
            foreach ($sampleStep in 5..20) {
                if ($sampleStep % 5 -eq 0) {
                    Write-Host "Sampling with step $sampleStep from checkpoint $checkpointFile..."
                    $outputDir = "samples/batch${batchSize}_lr${lr}_step${step}_sampleStep${sampleStep}"
                    New-Item -ItemType Directory -Force -Path $outputDir

                    & $pythonPath $sampleScript `
                        --model_path $checkpointFile `
                        --batch_size $batchSize `
                        --top_p -1 `
                        --out_dir $outputDir `
                        --seed 123 `
                        --step $sampleStep

                    # Evaluate
                    Write-Host "Evaluating samples in $outputDir..."
                    & $pythonPath $evalScript `
                        --folder $outputDir
                }
            }
        }
    }
}