import { Icon } from './Icon';

export function HowItWorks() {
  return (
    <section id="how-it-works" className="how" aria-labelledby="how-heading">
      <div className="container">
        <h2 id="how-heading" className="how__title">
          How this works
        </h2>
        <p className="how__intro">
          FrameCheck looks at two different kinds of evidence and keeps them apart, because they answer different
          questions.
        </p>
        <div className="how__grid">
          <article className="how__card">
            <span className="how__icon">
              <Icon name="shieldCheck" size={22} />
            </span>
            <h3>Provenance: what the file says about itself</h3>
            <p>
              Some cameras and editing apps attach <strong>Content Credentials</strong> — a signed record, based on the
              C2PA standard, of where an image came from and how it was changed. We check whether one is present and
              whether its signature holds up.
            </p>
            <p className="muted">
              A valid credential tells you who signed the record. A missing one is normal and tells you very little.
            </p>
          </article>
          <article className="how__card">
            <span className="how__icon">
              <Icon name="chart" size={22} />
            </span>
            <h3>Image analysis: what a model estimates</h3>
            <p>
              A detection model compares the image with patterns it learned from many examples and gives an estimate of
              how AI-like those patterns are. It’s a statistical guess, not a fact about how the image was made.
            </p>
            <p className="muted">
              Models can be fooled or confused by new generators, edits, screenshots and compression.
            </p>
          </article>
          <article className="how__card how__card--quiet">
            <span className="how__icon">
              <Icon name="layers" size={22} />
            </span>
            <h3>What neither can do</h3>
            <p>
              No single check — and no visual tell, like hands, text or lighting — can prove who or what made an image.
              Treat the result as one signal, and weigh it alongside where the image came from and who is sharing it.
            </p>
          </article>
        </div>
        <p className="how__prototype">
          <Icon name="info" size={16} />
          In this prototype, both checks are simulated with sample data. Your image is only read in your browser to show
          a preview and its dimensions.
        </p>
      </div>
    </section>
  );
}
