//! Minimal `.npy` reader (little-endian float32, C order).

pub fn read_f32(path: &std::path::Path) -> Result<(Vec<usize>, Vec<f32>), String> {
    let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if b.len() < 10 || &b[..6] != b"\x93NUMPY" {
        return Err("not a .npy file".into());
    }
    let (hlen, off) = if b[6] == 1 {
        (u16::from_le_bytes([b[8], b[9]]) as usize, 10)
    } else {
        (u32::from_le_bytes(b[8..12].try_into().unwrap()) as usize, 12)
    };
    let header = std::str::from_utf8(&b[off..off + hlen]).map_err(|e| e.to_string())?;
    if !header.contains("'descr': '<f4'") || header.contains("'fortran_order': True") {
        return Err(format!("unsupported npy header {header}"));
    }
    let s = header.find("'shape': (").ok_or("no shape")? + 10;
    let e = s + header[s..].find(')').ok_or("bad shape")?;
    let shape: Vec<usize> = header[s..e]
        .split(',')
        .map(str::trim)
        .filter(|x| !x.is_empty())
        .map(|x| x.parse().unwrap())
        .collect();
    let data = b[off + hlen..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    Ok((shape, data))
}
