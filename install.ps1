# Native, user-owned ROI Swarm Local bootstrap (Windows PowerShell 5.1+).
$ErrorActionPreference = 'Stop'
function Show-LocalHelp {
    Write-Output @'
roi-swarm-local: user-owned local inference, no provider keys
  --dir PATH / -Dir PATH (default: $HOME/roi-swarm-local)
  --backend auto|cuda|cpu / -Backend auto|cuda|cpu
  --profile balanced|fast / -Profile balanced|fast
  --with-security / -WithSecurity
  --accept-whiterabbitneo-terms / -AcceptWhiteRabbitNeoTerms
  --no-start / -NoStart
  --non-interactive / -NonInteractive
  --help / -Help
Requires Git for Windows. Run without Administrator privileges.
'@
}
function Assert-LocalPath([string]$Path) {
    $part = $Path
    while ($part) {
        if (Test-Path -LiteralPath $part) {
            $attributes = [IO.File]::GetAttributes($part)
            if (($attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Symlink/junction component refused: $part"
            }
        }
        $parent = [IO.Path]::GetDirectoryName($part)
        if ($parent -eq $part) { break }
        $part = $parent
    }
}
function Invoke-Checked([string]$Program, [string[]]$Arguments) {
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed (exit $LASTEXITCODE): $Program" }
}
function Install-LocalSwarm([object[]]$Options) {
    $root = Join-Path $HOME 'roi-swarm-local'
    $backend = 'auto'; $profile = 'balanced'
    $security = $false; $noStart = $false; $nonInteractive = $false
    $acceptTerms = $false
    for ($i = 0; $i -lt $Options.Count; $i++) {
        $option = [string]$Options[$i]
        switch -Regex ($option) {
            '^(--help|-Help|-h)$' { Show-LocalHelp; return }
            '^(--dir|-Dir|--backend|-Backend|--profile|-Profile)$' {
                if ($i + 1 -ge $Options.Count -or -not [string]$Options[$i + 1]) {
                    throw "Missing value for $option"
                }
                $i++; $value = [string]$Options[$i]
                switch -Regex ($option) {
                    '^(--dir|-Dir)$' { $root = $value }
                    '^(--backend|-Backend)$' { $backend = $value }
                    '^(--profile|-Profile)$' { $profile = $value }
                }
            }
            '^(--with-security|-WithSecurity)$' { $security = $true }
            '^(--accept-whiterabbitneo-terms|-AcceptWhiteRabbitNeoTerms)$' { $acceptTerms = $true }
            '^(--no-start|-NoStart)$' { $noStart = $true }
            '^(--non-interactive|-NonInteractive)$' { $nonInteractive = $true }
            default { throw "Unknown argument: $option (see --help)" }
        }
    }
    if ($backend -cnotin @('auto','cuda','cpu')) { throw '--backend must be auto, cuda or cpu' }
    if ($profile -cnotin @('balanced','fast')) { throw '--profile must be balanced or fast' }
    $principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
    if ($principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw 'Run PowerShell as your normal account, not as Administrator.'
    }
    $git = Get-Command git -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $git) { throw 'Required prerequisite: Git for Windows on PATH. Install it first, reopen PowerShell and retry.' }
    if (-not [IO.Path]::IsPathRooted($root) -or $root -match '[\r\n]' -or $root.StartsWith('\\')) {
        throw '--dir must be a local absolute path'
    }
    $root = [IO.Path]::GetFullPath($root).TrimEnd('\','/')
    if ($root -eq [IO.Path]::GetPathRoot($root).TrimEnd('\','/') -or $root -eq $HOME) {
        throw 'Unsafe installation path'
    }
    Assert-LocalPath $root
    $repo = 'https://github.com/Andrylynev/roi-swarm-local.git'
    $marker = Join-Path $root '.roi-swarm-local-install'
    $expected = "roi-swarm-local-v1`n$repo`n$root`n"
    $existing = $false
    if (Test-Path -LiteralPath $root) {
        if (-not (Test-Path -LiteralPath $root -PathType Container)) { throw 'Installation path is not a directory' }
        if (@(Get-ChildItem -LiteralPath $root -Force).Count -gt 0) {
            Assert-LocalPath $marker
            if (-not (Test-Path -LiteralPath $marker -PathType Leaf)) { throw 'Existing directory is unmarked; choose a new empty directory.' }
            if ([IO.File]::ReadAllText($marker) -cne $expected) { throw 'Installation ownership marker does not match this path/repository.' }
            $gitdir = Join-Path $root '.git'
            Assert-LocalPath $gitdir
            if (-not (Test-Path -LiteralPath $gitdir -PathType Container)) { throw 'Expected a standalone Git checkout' }
            $origin = @(& $git.Source -C $root config --local --get-all remote.origin.url)
            if ($LASTEXITCODE -ne 0 -or $origin.Count -ne 1 -or $origin[0] -cne $repo) { throw 'Untrusted origin; expected the public HTTPS repository.' }
            $top = & $git.Source -C $root rev-parse --show-toplevel
            if ($LASTEXITCODE -ne 0 -or [IO.Path]::GetFullPath($top) -ine $root) { throw 'Checkout root mismatch' }
            $existing = $true
        }
    }
    if ($existing) {
        $python = Join-Path $root '.venv/Scripts/python.exe'
        Assert-LocalPath $python
        if (Test-Path -LiteralPath $python -PathType Leaf) {
            Push-Location -LiteralPath $root
            try { Invoke-Checked $python @('-m','local.manage','stop') }
            catch { throw "Stop incomplete; refusing update/dependency reinstall. $_" }
            finally { Pop-Location }
        } elseif (Test-Path -LiteralPath (Join-Path $root 'data/local-services.json')) {
            throw 'Cannot prove services stopped: application interpreter is missing.'
        }
    }
    [IO.Directory]::CreateDirectory($root) | Out-Null
    if ($existing) {
        Invoke-Checked $git.Source @('-C',$root,'-c','core.hooksPath=NUL','-c','core.fsmonitor=false','fetch','--no-tags','origin','main')
        Invoke-Checked $git.Source @('-C',$root,'-c','core.hooksPath=NUL','-c','core.fsmonitor=false','merge','--ff-only','FETCH_HEAD')
    } else {
        Invoke-Checked $git.Source @('-c','core.hooksPath=NUL','clone','--branch','main','--',$repo,$root)
        [IO.File]::WriteAllText($marker, $expected, (New-Object Text.UTF8Encoding($false)))
        [IO.File]::AppendAllText((Join-Path $root '.git/info/exclude'), "`n/.roi-swarm-local-install`n")
    }
    $env:UV_CACHE_DIR = Join-Path $root 'cache/uv'
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $root 'tools/python'
    $env:UV_PROJECT_ENVIRONMENT = Join-Path $root '.venv'
    foreach ($name in @('VIRTUAL_ENV','CONDA_PREFIX','UV_PYTHON','PYTHONHOME','PYTHONPATH')) {
        Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue
    }
    $uvCommand = Get-Command uv -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($uvCommand) { $uv = $uvCommand.Source }
    else {
        $env:UV_INSTALL_DIR = Join-Path $root 'tools/uv'
        $env:UV_NO_MODIFY_PATH = '1'
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        $download = New-Object Net.WebClient
        try { $installer = $download.DownloadString('https://astral.sh/uv/install.ps1') }
        finally { $download.Dispose() }
        & ([ScriptBlock]::Create($installer))
        $uv = Join-Path $env:UV_INSTALL_DIR 'uv.exe'
        if (-not (Test-Path -LiteralPath $uv -PathType Leaf)) { throw 'uv user-scoped installation failed' }
    }
    Write-Output "Creating/updating application environment in $root"
    Push-Location -LiteralPath $root
    try {
        Invoke-Checked $uv @('sync','--frozen','--python','3.11')
        $setup = @('-m','local.setup','--backend',$backend,'--profile',$profile)
        if ($security) { $setup += '--with-security' }
        if ($acceptTerms) { $setup += '--accept-whiterabbitneo-terms' }
        if ($noStart) { $setup += '--no-start' }
        if ($nonInteractive) { $setup += '--non-interactive' }
        Invoke-Checked (Join-Path $root '.venv/Scripts/python.exe') $setup
    } finally { Pop-Location }
    Write-Output "ROI Swarm Local: $root"
    Write-Output 'Manage: .venv\Scripts\python.exe -m local.manage status (inside this directory)'
}
try { Install-LocalSwarm $args }
catch { Write-Error $_; exit 1 }
