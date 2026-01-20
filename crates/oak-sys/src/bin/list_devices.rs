use oak_sys::Device;

fn main() {
    let devices = Device::list();
    if devices.is_empty() {
        println!("No OAK devices found.");
        return;
    }

    println!("Found {} device(s):", devices.len());
    for dev in devices {
        println!(
            "- id: {}, name: {}, state: {:?}",
            dev.device_id, dev.name, dev.state
        );
    }
}
