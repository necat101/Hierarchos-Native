//! Keep companion image engines in the same cancellation lifetime as the CLI.
use std::process::Child;

#[cfg(windows)]
pub(super) struct ProcessTree(windows_sys::Win32::Foundation::HANDLE);
#[cfg(windows)]
impl ProcessTree {
    pub fn new(child:&Child)->std::io::Result<Self> {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::{Foundation::CloseHandle,System::JobObjects::*};
        unsafe {
            let job=CreateJobObjectW(std::ptr::null(),std::ptr::null());
            if job.is_null(){return Err(std::io::Error::last_os_error());}
            let mut limits:JOBOBJECT_EXTENDED_LIMIT_INFORMATION=std::mem::zeroed();
            limits.BasicLimitInformation.LimitFlags=JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            if SetInformationJobObject(job,JobObjectExtendedLimitInformation,&limits as *const _ as *const _,std::mem::size_of_val(&limits) as u32)==0||AssignProcessToJobObject(job,child.as_raw_handle())==0 {
                let error=std::io::Error::last_os_error();CloseHandle(job);return Err(error);
            }
            Ok(Self(job))
        }
    }
}
#[cfg(windows)]
impl Drop for ProcessTree {fn drop(&mut self){unsafe{windows_sys::Win32::Foundation::CloseHandle(self.0);}}}

#[cfg(unix)]
pub(super) struct ProcessTree(i32);
#[cfg(unix)]
impl ProcessTree {pub fn new(child:&Child)->std::io::Result<Self>{Ok(Self(child.id() as i32))}}
#[cfg(unix)]
impl Drop for ProcessTree {fn drop(&mut self){unsafe{libc::kill(-self.0,libc::SIGKILL);}}}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[cfg(windows)]
    fn closing_worker_job_terminates_its_process() {
        use std::os::windows::{io::AsRawHandle,process::CommandExt};
        use windows_sys::Win32::{Foundation::{WAIT_OBJECT_0,WAIT_TIMEOUT},System::Threading::WaitForSingleObject};
        let mut child=std::process::Command::new("cmd.exe").args(["/C","ping -n 30 127.0.0.1 >NUL"]).creation_flags(0x08000000).stdout(std::process::Stdio::null()).spawn().unwrap();
        let job=ProcessTree::new(&child).unwrap();
        assert_eq!(unsafe{WaitForSingleObject(child.as_raw_handle(),50)},WAIT_TIMEOUT);
        drop(job);
        assert_eq!(unsafe{WaitForSingleObject(child.as_raw_handle(),2000)},WAIT_OBJECT_0);
        let _=child.wait().unwrap();
    }
}
