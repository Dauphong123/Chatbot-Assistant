from .document import Document


class Chunker:
    def __init__(self, chunk_size: int = 300, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def recursive_chunk(self, document, separators=None):
        if separators is None:
            separators = ["\n\n", "\n", " ", ""]

        return self._split_text(document, separators)

    def _split_text(self, document, separators):
        """
        Recursive text splitter
        """
        text = document.text
        metadata = document.metadata
        separator = separators[-1]
        new_separators = []

        # base case
        if len(text) <= self.chunk_size:
            return [document]

        # Decide best separtors in the text right now
        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1 :]
                break

        # split text
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)

        final_chunks = []
        current_chunk = []
        current_length = 0

        for split in splits:
            s_len = len(split)
            if s_len == 0:
                continue
            # if the sum length is larger than the chunk -> push the chunk to final stack
            # clean the chunk till the chunk lenght less than the overlap_length and the sum length still not larger than chunk_size
            if (
                current_length + len(separator) + s_len > self.chunk_size
                and current_chunk
            ):
                chunk_text = separator.join(current_chunk)
                new_document = Document(chunk_text, metadata)
                final_chunks.extend(self._split_text(new_document, new_separators))

                while current_chunk and (
                    current_length > self.overlap
                    or (
                        current_length + len(separator) + s_len > self.chunk_size
                        and current_length > 0
                    )
                ):
                    removed = current_chunk.pop(0)
                    current_length -= len(removed)
                    if current_chunk:
                        current_length -= len(separator)

            # if the split bigger than the chunk_size than recursive_chunk the split
            if s_len > self.chunk_size:
                new_document = Document(split, metadata)
                final_chunks.extend(self._split_text(new_document, new_separators))
            else:
                current_length += s_len + len(separator) if current_chunk else s_len
                current_chunk.append(split)

        # after the loop if the chunk is not push to the final then do it
        if current_chunk:
            chunk_text = separator.join(current_chunk)
            new_document = Document(chunk_text, metadata)
            if len(chunk_text) > self.chunk_size:
                final_chunks.extend(self._split_text(new_document, new_separators))
            else:
                final_chunks.append(new_document)

        return final_chunks
