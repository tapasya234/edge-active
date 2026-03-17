import qai_hub


def share_with_organiers(
    job_id: str, organizer_email: str = "lowpowervision@gmail.com"
):
    compile_job = qai_hub.get_job(job_id)
    status = compile_job.get_status()

    if status.code == qai_hub.JobStatus.State.SUCCESS.name:
        print("Compilation successful")
    else:
        print(f"!!!! Compilation failed: {status.message}")
        return None

    compile_job.modify_sharing(add_emails=[organizer_email])

    print(f" Compile job shared: {compile_job.job_id}")


if __name__ == "__main__":
    share_with_organiers(job_id="jp83qkqxg")
