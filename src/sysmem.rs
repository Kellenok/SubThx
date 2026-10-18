//! Free memory of the machine, for choosing a batch size.

/// Free video memory of the adapter `index` (in DXGI order, software adapters skipped), in bytes.
/// With `nvidia` the NVIDIA adapter with the most memory is taken instead.
#[cfg(windows)]
pub fn free_vram(nvidia: bool) -> Option<u64> {
    if let Some(mib) = std::env::var("SUBTHX_FREE_VRAM_MIB").ok().and_then(|v| v.parse::<u64>().ok()) {
        return Some(mib << 20); // for trying what a smaller card would pick
    }
    use windows::core::Interface;
    use windows::Win32::Graphics::Dxgi::*;
    unsafe {
        let factory: IDXGIFactory1 = CreateDXGIFactory1().ok()?;
        let mut best: Option<(u64, u64)> = None; // (dedicated, free)
        let mut i = 0;
        while let Ok(adapter) = factory.EnumAdapters1(i) {
            i += 1;
            let Ok(desc) = adapter.GetDesc1() else { continue };
            if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0 {
                continue;
            }
            if nvidia && desc.VendorId != 0x10DE {
                continue;
            }
            let Ok(a3) = adapter.cast::<IDXGIAdapter3>() else { continue };
            let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
            if a3.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut info).is_err() {
                continue;
            }
            let free = info.Budget.saturating_sub(info.CurrentUsage);
            let dedicated = desc.DedicatedVideoMemory as u64;
            // DirectML's default device is the first adapter
            if !nvidia {
                return Some(free);
            }
            if best.is_none_or(|(d, _)| dedicated > d) {
                best = Some((dedicated, free));
            }
        }
        best.map(|(_, free)| free)
    }
}

#[cfg(not(windows))]
pub fn free_vram(_nvidia: bool) -> Option<u64> {
    std::env::var("SUBTHX_FREE_VRAM_MIB").ok().and_then(|v| v.parse::<u64>().ok()).map(|mib| mib << 20)
}

/// Physical memory the system can give to a program now, in bytes.
#[cfg(windows)]
pub fn available_ram() -> Option<u64> {
    use windows::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let mut st = MEMORYSTATUSEX { dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32, ..Default::default() };
    unsafe { GlobalMemoryStatusEx(&mut st).ok()? };
    Some(st.ullAvailPhys)
}

#[cfg(not(windows))]
pub fn available_ram() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let kb: u64 = text.lines().find_map(|l| l.strip_prefix("MemAvailable:"))?.trim().trim_end_matches("kB").trim().parse().ok()?;
    Some(kb * 1024)
}
