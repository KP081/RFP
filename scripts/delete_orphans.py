from app.api.deps import get_qdrant_service, get_rfp_repository

qdrant = get_qdrant_service()
known = {record.rfp_id for record in get_rfp_repository().list_all()}

# Qdrant me jitne rfp_id hain, sab nikalo (scroll = paginated read)
found: set[str] = set()
offset = None

while True:
    points, offset = qdrant.client.scroll(
        collection_name=qdrant.collection_name,
        with_payload=["rfp_id"],
        with_vectors=False,
        limit=500,
        offset=offset,
    )
    found.update(point.payload["rfp_id"] for point in points)

    if offset is None:
        break

orphans = found - known
print("In registry :", known)
print("In Qdrant   :", found)
print("Orphans     :", orphans)

for rfp_id in orphans:
    qdrant.delete_by_rfp(rfp_id)
    print("deleted", rfp_id)
