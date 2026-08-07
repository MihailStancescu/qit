## I've been trying to break attention

I've spent the last few months trying to break attention. I mean I've been trying to replace the mechanism itself. Not literally, obviously it still works fine in every transformer you've ever used.

Back in 2017, a handful of researchers published a paper with a title bold enough to become a meme: attention, they argued, was all you'd ever need. They were mostly right. Nearly every model built since decides what matters in a sentence the same way that paper described: take two tokens, multiply their vectors, and call the result "relevance." Dot products, softmax, done. It's a great trick. It's also the only trick anyone's really using.

So, I went looking for a different one, and I found it somewhere I didn't expect: inside a quantum circuit.

In quantum mechanics, when two signal paths meet, they don't get scored against each other and averaged. They interfere. Aligned phases reinforce each other. Opposed phases cancel out. Nothing gets compared, exactly. The answer just falls out of the physics.

That got me wondering whether a sequence model could work the same way. Instead of asking "how similar is token A to token B," what if a model just let every token's information collide and let the ones that agree amplify while the ones that don't fade out? No explicit similarity score at all, just a system where the right answer survives the interference and the wrong ones don't.

I built a small prototype to test it. Encoded tokens as rotations, entangled them, ran the whole thing through a circuit that plays the role attention normally plays in a transformer, and gave it a task specifically chosen because it has almost no shortcut for a normal neural network to exploit.

I wasn't expecting much from a first pass. What I got back made me sit and stare at the screen for a while.

I'm not going to share numbers yet. I'm writing this up properly, and I'd rather you see the real result in context than as a stat in a LinkedIn post. But I'll say this much: it converged faster than I thought was reasonable, with a fraction of the parameters I expected to need, and it did it by leaning on exactly the kind of structure quantum interference is good at.

Paper is on the way. If any of these topics overlaps with what you work on, follow along. I'd rather hear the hard questions now than after it's published.

Interference computes. I'm still finding out what else it can learn.

---

**Image prompt (for the post's header image):**

Abstract visualization of overlapping wave patterns in deep blues and purples, some waves reinforcing into bright bands of light where they align, others cancelling into dark gaps where they oppose, set against a black background. Faint circuit-like lines run through the waves, suggesting a quantum processor without showing literal hardware. No text, no logos, no diagrams, no readable data. Clean, moody, high-contrast, minimal, editorial tech aesthetic, wide banner ratio suitable for a LinkedIn post header.

**Optional second visual: a qualitative convergence curve**

A simple line chart with generic, unitless axis labels only ("Training steps" on the x-axis, "Performance" on the y-axis), no tick marks, no numbers, and no legend, just two curves on the same plot. One curve drops toward the bottom of the chart within the first few steps and flattens out. The other descends slowly and gradually over a much longer stretch before it flattens. The shapes alone should suggest "one of these got there faster" without giving away epochs, accuracy, or parameter counts. Clean minimal chart style, dark background to match the header image, no title.

Suggested caption for this image: "Two training runs, same task. I'll leave you to guess which curve is the new approach."