from app.api.deps import get_rfp_service

service = get_rfp_service()

for rfp in service.list_all():
    print(f"Re-ingesting {rfp.filename} ({rfp.rfp_id}) ...")
    service.reingest(rfp.rfp_id)
    service.process(rfp.rfp_id)

    updated = service.get(rfp.rfp_id)
    print(
        f"  -> {updated.status} | pages={updated.pages} chunks={updated.chunks} {updated.error or ''}"
    )
