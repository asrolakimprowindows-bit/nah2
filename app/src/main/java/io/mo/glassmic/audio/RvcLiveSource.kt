package io.mo.glassmic.audio

import io.mo.glassmic.core.model.SourceType
import java.nio.ByteBuffer
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.atomic.AtomicInteger

/**
 * Live AudioSourceProvider fed by an external pipeline (in our case: mic → Kaggle RVC →
 * decoded PCM) instead of a static file or pre-generated TTS buffer.
 *
 * Unlike [BufferedPcmSource] (one fixed byte[] known up front), this accepts new PCM16
 * chunks pushed in continuously via [push] while GlassMic's broadcast loop keeps calling
 * [read] on its own steady real-time cadence. Returning 0 (not -1) when the queue is
 * momentarily empty is what keeps this from being treated as EOF — the broadcast loop
 * just retries a couple ms later, exactly the same as it already does for a slow file
 * read, so a network round-trip gap plays as a brief pause rather than ending playback.
 *
 * PCM pushed in via [push] MUST already be 48kHz mono PCM16 (GlassMic's master format).
 * Use [Resampler.to48kMono16] to convert our 44100Hz recorder output before pushing.
 */
class RvcLiveSource : AudioSourceProvider {

    override val type = SourceType.TTS // no dedicated RVC case yet; TTS plays the same "external live buffer" role

    private val queue = ConcurrentLinkedQueue<Byte>()
    private val queuedBytes = AtomicInteger(0)
    @Volatile private var stopped = false

    /** Called from our RVC pipeline whenever a converted chunk comes back from Kaggle. */
    fun push(pcm16At48kMono: ByteArray) {
        if (stopped) return
        for (b in pcm16At48kMono) queue.add(b)
        queuedBytes.addAndGet(pcm16At48kMono.size)
    }

    /** Bytes still waiting to be played — useful for a simple "buffering…" UI indicator. */
    fun pendingBytes(): Int = queuedBytes.get()

    override suspend fun read(out: ByteBuffer, sampleRate: Int, channels: Int): Int {
        if (stopped) return -1
        val want = out.remaining()
        var written = 0
        while (written < want) {
            val b = queue.poll() ?: break
            out.put(b)
            written++
        }
        if (written > 0) queuedBytes.addAndGet(-written)
        return written // 0 is fine here — publisher just retries, does NOT mean EOF
    }

    /** Call when the RVC session ends so the broadcast loop can fall back to silence/real mic. */
    fun stop() {
        stopped = true
        queue.clear()
        queuedBytes.set(0)
    }

    override fun reset() {
        queue.clear()
        queuedBytes.set(0)
        stopped = false
    }

    override fun release() = stop()
}

/** Simple linear-interpolation resampler — no external deps, good enough for speech. */
object Resampler {

    /** Converts 16-bit little-endian PCM at [srcSampleRate] mono to 48kHz mono PCM16LE. */
    fun to48kMono16(input: ByteArray, srcSampleRate: Int): ByteArray {
        if (srcSampleRate == MASTER_SAMPLE_RATE) return input
        val inSamples = input.size / 2
        if (inSamples <= 0) return ByteArray(0)
        val ratio = MASTER_SAMPLE_RATE.toDouble() / srcSampleRate.toDouble()
        val outSamples = (inSamples * ratio).toInt().coerceAtLeast(1)
        val out = ByteArray(outSamples * 2)
        for (j in 0 until outSamples) {
            val srcPos = j / ratio
            val i0 = srcPos.toInt().coerceIn(0, inSamples - 1)
            val i1 = (i0 + 1).coerceAtMost(inSamples - 1)
            val frac = srcPos - i0
            val s0 = sampleAt(input, i0)
            val s1 = sampleAt(input, i1)
            val v = (s0 + (s1 - s0) * frac).toInt().coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt())
            out[j * 2] = (v and 0xFF).toByte()
            out[j * 2 + 1] = ((v ushr 8) and 0xFF).toByte()
        }
        return out
    }

    /** Converts a 32-bit float PCM array (what our RvcAudioEncoder decodes to) straight to
     *  48kHz mono PCM16LE bytes, resampling from [srcSampleRate] in the same pass. */
    fun floatArrayTo48kMono16(input: FloatArray, srcSampleRate: Int): ByteArray {
        val inSamples = input.size
        if (inSamples <= 0) return ByteArray(0)
        val ratio = MASTER_SAMPLE_RATE.toDouble() / srcSampleRate.toDouble()
        val outSamples = (inSamples * ratio).toInt().coerceAtLeast(1)
        val out = ByteArray(outSamples * 2)
        for (j in 0 until outSamples) {
            val srcPos = j / ratio
            val i0 = srcPos.toInt().coerceIn(0, inSamples - 1)
            val i1 = (i0 + 1).coerceAtMost(inSamples - 1)
            val frac = (srcPos - i0).toFloat()
            val s0 = input[i0]
            val s1 = input[i1]
            val f = s0 + (s1 - s0) * frac
            val v = (f * 32767f).toInt().coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt())
            out[j * 2] = (v and 0xFF).toByte()
            out[j * 2 + 1] = ((v ushr 8) and 0xFF).toByte()
        }
        return out
    }

    private fun sampleAt(b: ByteArray, idx: Int): Int =
        ((b[idx * 2 + 1].toInt() shl 8) or (b[idx * 2].toInt() and 0xFF)).toShort().toInt()

    const val MASTER_SAMPLE_RATE = 48_000
}
