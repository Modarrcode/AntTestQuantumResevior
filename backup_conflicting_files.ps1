Set-Location 'd:/Masters Project/AntTestQuantumResevior'
$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$cwd = (Get-Location).Path
$bk = $cwd + '\backup_before_merge_' + $ts
New-Item -ItemType Directory -Path $bk -Force | Out-Null
$files = @(
 'AntWithRes.py',
 'improved_rc_model/summary.txt',
 'multifunctional_rc_ae_model/excel_episode_results.csv',
 'multifunctional_rc_ae_model/excel_kpi_overall.csv',
 'multifunctional_rc_ae_model/excel_kpi_results.csv',
 'multifunctional_rc_ae_model/excel_kpi_robust.csv',
 'multifunctional_rc_ae_model/excel_paper_table.csv',
 'multifunctional_rc_ae_model/excel_summary_results.csv',
 'multifunctional_rc_ae_model/summary.txt',
 'multifunctional_rc_autoencoder/summary.txt',
 'rc_rl_extended/extended_rl_run.log',
 'simple_rc_model/summary.txt',
 'single_rc_model/summary.txt',
 'test_imports.py'
)
foreach($f in $files){
    if(Test-Path $f){
        $dest = Join-Path $bk $f
        $destDir = Split-Path $dest -Parent
        if(-not(Test-Path $destDir)){
            New-Item -ItemType Directory -Path $destDir -Force | Out-Null
        }
        Move-Item -Path $f -Destination $dest -Force
    }
}
Write-Output ('Backed up conflicting files to: ' + $bk)
