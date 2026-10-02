import numpy as np

from app.face_registry import FaceRegistry


registry = FaceRegistry(
    similarity_threshold=0.45
)


# Simulated face embedding
embedding_1 = np.array([
    0.1,
    0.2,
    0.3,
    0.4
])


# Very similar to embedding_1
embedding_2 = np.array([
    0.1,
    0.2,
    0.3,
    0.41
])


# Completely different example
embedding_3 = np.array([
    -0.8,
    0.1,
    -0.2,
    0.7
])


face_id, is_new, similarity = registry.identify_or_register(
    embedding_1
)

print("First face:")
print("Face ID:", face_id)
print("New:", is_new)
print("Similarity:", similarity)


face_id, is_new, similarity = registry.identify_or_register(
    embedding_2
)

print("\nSecond face:")
print("Face ID:", face_id)
print("New:", is_new)
print("Similarity:", similarity)


face_id, is_new, similarity = registry.identify_or_register(
    embedding_3
)

print("\nThird face:")
print("Face ID:", face_id)
print("New:", is_new)
print("Similarity:", similarity)


print("\nTotal registered faces:")
print(len(registry.registered_faces))